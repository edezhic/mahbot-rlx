// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Networked multi-node transport (plan #12/#49, multi-node).
//!
//! [`NetTransport`] is one TCP connection per rank pair (full mesh)
//! with a single reader thread per connection that *demultiplexes*
//! every incoming frame into the right place:
//!
//!   - `SEND`    → a two-sided inbox keyed by `(src, tag)` that
//!     [`Transport::recv_bytes`] pops (pipeline-parallel hidden states).
//!   - `PUT`     → this rank's symmetric heap (one-sided write).
//!   - `GETREQ`  → read this rank's heap, reply with `GETRESP`.
//!   - `GETRESP` → a response inbox the blocked
//!     [`SymmetricTransport::get`] pops.
//!
//! Because the reader thread services the socket, a blocked `recv`/`get`
//! on the main thread never stalls the peer — so the gather-to-root
//! collectives in [`crate::transport::ProcessGroup`] (and the
//! one-sided collectives in [`crate::collective`]) are deadlock-free
//! over any rank count.
//!
//! Two public constructors expose the same engine with different
//! intent:
//!   - [`TcpTransport::bind`] — portable; give it any reachable IPs
//!     (Ethernet, or the macOS Thunderbolt Bridge link).
//!   - [`ThunderboltTransport::bind`] — same wire protocol, intended
//!     for IPs on the Thunderbolt interface, and the place a future
//!     zero-copy DMA backend slots in behind the unchanged
//!     [`Transport`] + [`SymmetricTransport`] traits.

use crate::symmetric::{CollectiveError, Rank, SymmetricBuffer, SymmetricTransport};
use crate::transport::{Transport, default_barrier};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// Frame kinds. Header is fixed 13 bytes: [kind:u8][a:u32 le][b:u32 le][len:u32 le].
const HELLO: u8 = 1; // a = connector rank, no payload
const SEND: u8 = 2; // a = src rank, b = tag, payload = bytes
const PUT: u8 = 3; // a = src rank, b = offset, payload = bytes to write
const GETREQ: u8 = 4; // a = requester rank, payload = [offset:u32 le][len:u32 le]
const GETRESP: u8 = 5; // a = responder rank, payload = bytes

const HEADER_LEN: usize = 13;

fn write_frame<W: Write>(w: &mut W, kind: u8, a: u32, b: u32, payload: &[u8]) -> io::Result<()> {
    let mut hdr = [0u8; HEADER_LEN];
    hdr[0] = kind;
    hdr[1..5].copy_from_slice(&a.to_le_bytes());
    hdr[5..9].copy_from_slice(&b.to_le_bytes());
    hdr[9..13].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    // Coalesce header + payload into ONE write for the small frames that
    // dominate collective latency: with TCP_NODELAY set, two separate
    // writes became two segments and an extra RTT-coupled syscall. For
    // large frames the second write is negligible against the payload, so
    // keep them separate and skip the copy.
    const COALESCE_MAX: usize = 1 << 16;
    if payload.len() <= COALESCE_MAX {
        let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
        w.write_all(&buf)?;
    } else {
        w.write_all(&hdr)?;
        w.write_all(payload)?;
    }
    w.flush()
}

/// Decode a frame header: `(kind, a, b, payload length)`.
fn parse_header(hdr: &[u8; HEADER_LEN]) -> (u8, u32, u32, u32) {
    (
        hdr[0],
        u32::from_le_bytes(hdr[1..5].try_into().unwrap()),
        u32::from_le_bytes(hdr[5..9].try_into().unwrap()),
        u32::from_le_bytes(hdr[9..13].try_into().unwrap()),
    )
}

fn read_frame<R: Read>(r: &mut R) -> io::Result<(u8, u32, u32, Vec<u8>)> {
    let mut hdr = [0u8; HEADER_LEN];
    r.read_exact(&mut hdr)?;
    let (kind, a, b, len) = parse_header(&hdr);
    let mut payload = vec![0u8; len as usize];
    if len > 0 {
        r.read_exact(&mut payload)?;
    }
    Ok((kind, a, b, payload))
}

/// `RLX_CONNECT_WAIT` seconds: how long a dial to a peer keeps trying, and how
/// long the acceptor waits for the ranks it expects to greet. `0` (`None`) is no
/// limit on either — a persistent node waits for the next job's rank 0 as long
/// as it takes. 10 s by default.
fn env_connect_wait() -> Option<Duration> {
    match rlx_ir::env::read("RLX_CONNECT_WAIT")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        Some(s) if s <= 0.0 => None,
        Some(s) => Some(Duration::from_secs_f64(s)),
        None => Some(Duration::from_secs(10)),
    }
}

/// The longest one connection attempt may take. A peer whose SYNs go
/// unanswered (a Wi-Fi stall, a dropped route) otherwise blocks a plain
/// `connect` for over a minute each try, and a "10 s" budget became a mesh
/// that hung for half an hour.
const CONNECT_ATTEMPT: Duration = Duration::from_secs(2);

/// How long the acceptor sleeps between non-blocking `accept` calls once the mesh
/// has formed, so `Drop` never waits on a blocked one — and so this is the longest
/// `Drop` waits for the acceptor to notice `shutdown`. The loop runs for the
/// transport's life (that is what takes a genuine peer that dials after a stale
/// greeting), so this is also its idle wakeup rate — the rate once no rank is
/// outstanding; while one is, the loop looks far sooner. A sleeping thread between
/// ticks costs nothing.
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// How long an accepted connection has to say `HELLO` before it is dropped.
const HELLO_WAIT: Duration = Duration::from_secs(10);

/// How many connections may wait for their `HELLO` at once.
const MAX_SILENT: usize = 32;

fn connect_retry(addr: SocketAddr) -> io::Result<TcpStream> {
    let deadline = env_connect_wait().map(|w| Instant::now() + w);
    let mut last: Option<io::Error> = None;
    loop {
        let left = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if left.is_some_and(|l| l.is_zero()) {
            break;
        }
        let attempt = left.map_or(CONNECT_ATTEMPT, |l| l.min(CONNECT_ATTEMPT));
        match TcpStream::connect_timeout(&addr, attempt.max(Duration::from_millis(10))) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last = Some(e);
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
    let e = last.unwrap_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "no attempt made"));
    Err(io::Error::new(e.kind(), format!("connect to {addr}: {e}")))
}

struct NetInner {
    rank: u32,
    world: u32,
    heap_size: usize,
    /// Write half of the connection to each peer (index = peer rank).
    /// `None` at our own rank and until a peer connects.
    writers: Vec<Mutex<Option<TcpStream>>>,
    /// Two-sided inbox: `(src_rank, tag) -> queued payloads`.
    inbox: Mutex<HashMap<(u32, u32), VecDeque<Vec<u8>>>>,
    inbox_cv: Condvar,
    /// This rank's symmetric heap (target of remote PUT / GETREQ).
    heap: RwLock<Vec<u8>>,
    /// Responses to our one-sided GETs: `responder_rank -> payloads`.
    getresp: Mutex<HashMap<u32, VecDeque<Vec<u8>>>>,
    getresp_cv: Condvar,
    shutdown: AtomicBool,
    /// Per peer: `1 +` the generation of the connection last seen to end (`0`:
    /// none). A receive waiting on a lost peer fails instead of blocking forever,
    /// and a peer is lost only while the live generation is the one that died.
    lost: Vec<AtomicU64>,
    /// Per peer: which of its connections is the current one. A reader whose
    /// connection was replaced (a later HELLO from the same rank) ends
    /// without marking the peer lost.
    conn_gen: Vec<AtomicU64>,
    /// Upper bound on one receive (`RLX_COLLECTIVE_TIMEOUT` seconds).
    recv_timeout: Option<Duration>,
}

/// `RLX_COLLECTIVE_TIMEOUT` (seconds), read once per transport.
/// `RLX_MESH_REVERSE="0-3,0-4"`: pairs of ranks whose connection the
/// higher rank opens — the lower one cannot reach it (an iPad accepts no
/// connection over its USB-C link, but can open one). Every rank of the
/// mesh must be given the same list.
fn env_mesh_reverse() -> Vec<(u32, u32)> {
    rlx_ir::env::read("RLX_MESH_REVERSE")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|p| {
                    let (a, b) = p.trim().split_once('-')?;
                    let (a, b) = (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?);
                    Some((a.min(b), a.max(b)))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn env_recv_timeout() -> Option<Duration> {
    rlx_ir::env::read("RLX_COLLECTIVE_TIMEOUT")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map(Duration::from_secs)
}

impl NetInner {
    fn send_frame(
        &self,
        peer: u32,
        kind: u8,
        a: u32,
        b: u32,
        payload: &[u8],
    ) -> Result<(), CollectiveError> {
        let mut guard = self.writers[peer as usize].lock().unwrap();
        match guard.as_mut() {
            Some(s) => {
                write_frame(s, kind, a, b, payload).map_err(|e| CollectiveError::TransportError {
                    reason: format!("send to rank {peer}: {e}"),
                })
            }
            None => Err(CollectiveError::TransportError {
                reason: format!("no connection to rank {peer}"),
            }),
        }
    }

    /// Blocking receive. Queued data is always delivered first; after that a
    /// lost peer, or `recv_timeout` passing, is an error that names the rank —
    /// so a dead node stops the job with a reason instead of hanging it.
    fn recv_inbox(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
        if from >= self.world {
            return Err(CollectiveError::TransportError {
                reason: format!("rank {from} out of range for WORLD={}", self.world),
            });
        }
        let deadline = self.recv_timeout.map(|t| Instant::now() + t);
        let mut guard = self.inbox.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return Ok(v);
            }
            if from != self.rank && self.writers[from as usize].lock().unwrap().is_none() {
                return Err(CollectiveError::TransportError {
                    reason: format!(
                        "rank {} has no connection to rank {from} (a star topology links \
                         workers only to rank 0)",
                        self.rank
                    ),
                });
            }
            if self.is_lost(from) {
                return Err(CollectiveError::TransportError {
                    reason: format!(
                        "rank {from} disconnected (process exited or host unreachable) \
                         while rank {} waited on tag {tag}",
                        self.rank
                    ),
                });
            }
            match deadline {
                None => guard = self.inbox_cv.wait(guard).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Err(CollectiveError::TransportError {
                            reason: format!(
                                "rank {} waited {}s for rank {from} (tag {tag}) — \
                                 RLX_COLLECTIVE_TIMEOUT",
                                self.rank,
                                self.recv_timeout.unwrap_or_default().as_secs()
                            ),
                        });
                    }
                    guard = self.inbox_cv.wait_timeout(guard, d - now).unwrap().0;
                }
            }
        }
    }

    fn is_lost(&self, peer: u32) -> bool {
        let i = peer as usize;
        let Some((lost, current)) = self.lost.get(i).zip(self.conn_gen.get(i)) else {
            return false;
        };
        let lost = lost.load(Ordering::SeqCst);
        lost != 0 && lost - 1 == current.load(Ordering::SeqCst)
    }

    /// The reader of `mine` saw `peer`'s connection end: mark that connection lost
    /// and wake every waiter, so receives on it fail now. A later connection has a
    /// later generation, so this neither needs a clear nor can mask the loss of one
    /// that died after it.
    fn mark_lost(&self, peer: u32, mine: u64) {
        if let Some(lost) = self.lost.get(peer as usize) {
            lost.fetch_max(mine + 1, Ordering::SeqCst);
        }
        // Take the lock so a waiter between its check and `wait` sees this.
        drop(self.inbox.lock().unwrap());
        self.inbox_cv.notify_all();
        self.getresp_cv.notify_all();
    }

    /// Like [`Self::recv_inbox`] but returns `None` if nothing matching
    /// `(from, tag)` arrives before `timeout` elapses. Lets a collective
    /// bound how long it waits on a slow/offline peer (bounded staleness).
    fn recv_inbox_timeout(&self, from: u32, tag: u32, timeout: Duration) -> Option<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.inbox.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return Some(v);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let (g, res) = self.inbox_cv.wait_timeout(guard, deadline - now).unwrap();
            guard = g;
            if res.timed_out()
                && let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return Some(v);
            }
        }
    }

    fn wait_getresp(&self, peer: u32) -> Vec<u8> {
        let mut guard = self.getresp.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&peer)
                && let Some(v) = q.pop_front()
            {
                return v;
            }
            guard = self.getresp_cv.wait(guard).unwrap();
        }
    }

    fn check_buf(&self, buf: SymmetricBuffer) -> Result<(), CollectiveError> {
        if buf.rank.0 >= self.world {
            return Err(CollectiveError::UnknownRank {
                rank: buf.rank,
                num_ranks: self.world,
            });
        }
        if buf.offset + buf.len > self.heap_size {
            return Err(CollectiveError::OutOfBounds {
                rank: buf.rank,
                offset: buf.offset,
                len: buf.len,
                heap_size: self.heap_size,
            });
        }
        Ok(())
    }

    fn dispatch(&self, kind: u8, a: u32, b: u32, payload: Vec<u8>) {
        match kind {
            SEND => {
                self.inbox
                    .lock()
                    .unwrap()
                    .entry((a, b))
                    .or_default()
                    .push_back(payload);
                self.inbox_cv.notify_all();
            }
            PUT => {
                let off = b as usize;
                if off + payload.len() <= self.heap_size {
                    let mut h = self.heap.write().unwrap();
                    h[off..off + payload.len()].copy_from_slice(&payload);
                }
            }
            GETREQ if payload.len() >= 8 => {
                let off = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                let rlen = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
                let data = {
                    let h = self.heap.read().unwrap();
                    if off + rlen <= self.heap_size {
                        h[off..off + rlen].to_vec()
                    } else {
                        Vec::new()
                    }
                };
                // a = requester rank; respond with our rank as sender.
                let _ = self.send_frame(a, GETRESP, self.rank, off as u32, &data);
            }
            GETRESP => {
                self.getresp
                    .lock()
                    .unwrap()
                    .entry(a)
                    .or_default()
                    .push_back(payload);
                self.getresp_cv.notify_all();
            }
            _ => {}
        }
    }
}

/// Read `peer`'s frames until its connection closes. `mine` is this
/// connection's generation ([`NetInner::conn_gen`]), taken when it was
/// accepted, not when this thread starts.
fn reader_loop(mut stream: TcpStream, inner: Arc<NetInner>, peer: u32, mine: u64) {
    let current = |inner: &NetInner| inner.conn_gen[peer as usize].load(Ordering::SeqCst);
    // Exits when read_frame errors (peer closed or we were shut down).
    while let Ok((kind, a, b, payload)) = read_frame(&mut stream) {
        if current(&inner) != mine {
            break;
        }
        inner.dispatch(kind, a, b, payload);
        if inner.shutdown.load(Ordering::Relaxed) {
            break;
        }
    }
    if !inner.shutdown.load(Ordering::Relaxed) {
        inner.mark_lost(peer, mine);
    }
}

/// A connection the acceptor has taken but that has not said `HELLO` yet.
struct Silent {
    stream: TcpStream,
    addr: SocketAddr,
    /// When it was accepted: one that never speaks is dropped after `HELLO_WAIT`.
    since: Instant,
}

/// The constructor's wait on the ranks it expects: when it gives up, and the
/// budget it reports (see `env_connect_wait`).
struct GiveUp {
    at: Instant,
    budget: Duration,
}

/// The ranks a mesh accepts from, which of them have still to greet, and the
/// constructor's bounded wait on the last of them.
struct Greetings {
    /// Every rank we accept from, greeted or not: a repeat `HELLO` must still be
    /// taken, so a connection is checked against this set and not against
    /// `missing`.
    expect: BTreeSet<u32>,
    /// The ranks that have not greeted yet.
    missing: Mutex<BTreeSet<u32>>,
    cv: Condvar,
    /// `None` is no limit (a persistent node waiting for the next job).
    give_up: Option<GiveUp>,
}

impl Greetings {
    fn new(expect: BTreeSet<u32>, budget: Option<Duration>) -> Self {
        Self {
            missing: Mutex::new(expect.clone()),
            // A poll of slack: the acceptor credits a greeting on its next look
            // rather than the instant it lands, so one that arrives just inside
            // the budget still counts.
            give_up: budget.map(|budget| GiveUp {
                at: Instant::now() + budget + ACCEPT_POLL,
                budget,
            }),
            expect,
            cv: Condvar::new(),
        }
    }

    /// Is `peer` a rank we accept from?
    fn expects(&self, peer: u32) -> bool {
        self.expect.contains(&peer)
    }

    /// `peer` greeted; `false` if it had already greeted (a repeat).
    fn greeted(&self, peer: u32) -> bool {
        let first = self.missing.lock().unwrap().remove(&peer);
        self.cv.notify_all();
        first
    }

    /// Is the constructor still waiting for a rank that can still arrive? False
    /// once every expected rank has greeted, and false once the budget has run
    /// out — after that the acceptor only has a late dial left to catch.
    fn waiting(&self) -> bool {
        self.waiting_on(&self.missing.lock().unwrap())
    }

    /// [`Self::waiting`]'s rule, for a caller that already holds `missing`'s lock.
    fn waiting_on(&self, missing: &BTreeSet<u32>) -> bool {
        !missing.is_empty() && self.give_up.as_ref().is_none_or(|g| Instant::now() < g.at)
    }

    /// Wait until every expected rank has greeted, or the budget runs out.
    fn wait(&self, rank: u32) -> io::Result<()> {
        let mut missing = self.missing.lock().unwrap();
        while self.waiting_on(&missing) {
            missing = match &self.give_up {
                Some(g) => {
                    self.cv
                        .wait_timeout(missing, g.at.saturating_duration_since(Instant::now()))
                        .unwrap()
                        .0
                }
                None => self.cv.wait(missing).unwrap(),
            };
        }
        match &self.give_up {
            // Ranks still missing: the loop can only have left on the budget.
            Some(g) if !missing.is_empty() => {
                let who: Vec<String> = missing.iter().map(|p| format!("rank {p}")).collect();
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "rank {rank} waited {}s for {} to greet — RLX_CONNECT_WAIT",
                        g.budget.as_secs_f64(),
                        who.join(", ")
                    ),
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Accept the ranks that dial us, for as long as the transport lives.
///
/// A rank's slot is filled by the last `HELLO` heard for it, and accepting does
/// not stop once every expected rank has greeted: a persistent worker from the
/// run before, still dialing until it is stopped, announces a rank that its new
/// process announces again, and the genuine connection has to be taken when it
/// arrives even though the slot was already filled. A connection that is not a
/// `HELLO` from a rank we expect is dropped.
///
/// Nothing a peer does holds the loop: a connection that has not said `HELLO`
/// yet waits in `silent` instead of being read where it stands, so the
/// connections queued behind it are still taken, and the wait the constructor
/// bounds with `RLX_CONNECT_WAIT` is not spent on a peer that never speaks. While
/// `silent` is full, acceptance pauses and the kernel backlog holds the rest,
/// rather than dropping a connection that may be a genuine peer — one that never
/// re-dials. A silent peer frees its slot after `HELLO_WAIT`, so a flood that
/// fills the list costs a delay of up to `HELLO_WAIT` — the default budget, so it
/// can still fail the mesh naming the rank it held up — but it never drops one.
///
/// Excluded, both because the wire carries no session identity and a slot keeps the
/// last greeting heard: a stale greeting that fills the slot before the genuine peer
/// dials leaves the constructor returning with that slot about to be replaced, and a
/// stale peer that dials *after* the genuine one takes the slot from it. What either
/// costs a caller is stated on [`NetTransport::from_listener`]. A write already in
/// flight to a peer that stopped reading holds that slot's lock until it returns
/// (writes have no timeout), so its replacement waits for it.
fn accept_loop(
    listener: TcpListener,
    inner: Arc<NetInner>,
    readers: Arc<Mutex<Vec<JoinHandle<()>>>>,
    greetings: Arc<Greetings>,
) {
    let rank = inner.rank;
    // Connections that have not said `HELLO` yet, oldest first: one that never
    // speaks is dropped after `HELLO_WAIT`.
    let mut silent: Vec<Silent> = Vec::new();
    // Trouble is reported once rather than on every tick — see `report_once`.
    let mut quiet = false;
    while !inner.shutdown.load(Ordering::Relaxed) {
        // A tick that took a connection goes straight on to the next one, so a
        // mesh forms without waiting out a poll; only an idle tick sleeps. That is
        // bounded by the connections queued, so this cannot spin.
        let mut busy = false;
        // Take what is queued, up to the cap.
        while silent.len() < MAX_SILENT {
            match listener.accept() {
                Ok((stream, addr)) => {
                    stream.set_nodelay(true).ok();
                    if stream.set_nonblocking(true).is_err() {
                        continue;
                    }
                    silent.push(Silent {
                        stream,
                        addr,
                        since: Instant::now(),
                    });
                    busy = true;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    report_once(
                        &mut quiet,
                        rank,
                        format_args!("accept failed, retrying: {e}"),
                    );
                    break;
                }
            }
        }
        let mut i = 0;
        while i < silent.len() {
            let frame = match header_waiting(&silent[i].stream) {
                Ok(Some(frame)) => frame,
                Ok(None) if silent[i].since.elapsed() < HELLO_WAIT => {
                    i += 1;
                    continue;
                }
                Ok(None) => {
                    let addr = silent.remove(i).addr;
                    report_once(
                        &mut quiet,
                        rank,
                        format_args!(
                            "dropped a connection from {addr}: no HELLO in {HELLO_WAIT:?}"
                        ),
                    );
                    continue;
                }
                Err(e) => {
                    let addr = silent.remove(i).addr;
                    report_once(
                        &mut quiet,
                        rank,
                        format_args!("dropped a connection from {addr}: {e}"),
                    );
                    continue;
                }
            };
            // The whole frame is here: a `HELLO` is a header and no payload, so
            // it is judged before anything is consumed.
            let (kind, peer, _, len) = frame;
            if kind != HELLO || len != 0 || !greetings.expects(peer) {
                let addr = silent.remove(i).addr;
                report_once(
                    &mut quiet,
                    rank,
                    format_args!(
                        "dropped a connection from {addr}: kind {kind}, rank {peer}, {len} bytes"
                    ),
                );
                continue;
            }
            // A write to this peer may be in flight and blocked on a peer that
            // stopped reading. Take the slot if it is free and come back next
            // tick otherwise, rather than wait behind that write.
            let Ok(mut slot) = inner.writers[peer as usize].try_lock() else {
                i += 1;
                continue;
            };
            let Silent {
                mut stream, addr, ..
            } = silent.remove(i);
            // Consume the frame the peek showed, so the reader starts at the
            // next one.
            let mut consumed = [0u8; HEADER_LEN];
            if stream.set_nonblocking(false).is_err() || stream.read_exact(&mut consumed).is_err() {
                report_once(
                    &mut quiet,
                    rank,
                    format_args!("dropped a connection from {addr}: unreadable HELLO"),
                );
                continue;
            }
            let (rd, wr) = match (stream.try_clone(), stream.try_clone()) {
                (Ok(rd), Ok(wr)) => (rd, wr),
                (Err(e), _) | (_, Err(e)) => {
                    report_once(
                        &mut quiet,
                        rank,
                        format_args!("dropped rank {peer} from {addr}: {e}"),
                    );
                    continue;
                }
            };
            // The slot is filled before the generation is published, so whoever
            // sees the new generation also sees the connection it belongs to.
            let old = slot.replace(wr);
            let gen_ = inner.conn_gen[peer as usize].fetch_add(1, Ordering::SeqCst) + 1;
            drop(slot);
            if let Some(old) = old {
                let _ = old.shutdown(Shutdown::Both);
            }
            let inner2 = inner.clone();
            let reader = thread::spawn(move || reader_loop(rd, inner2, peer, gen_));
            let mut handles = readers.lock().unwrap();
            // A replaced connection's reader is done with; only the live ones
            // need joining, and this keeps the list from growing for the
            // transport's life.
            handles.retain(|h| !h.is_finished());
            handles.push(reader);
            drop(handles);
            // Only now is the slot usable, so only now is the rank no longer one
            // the constructor waits for.
            if greetings.greeted(peer) {
                // A first greeting landed: whatever was being reported is over.
                quiet = false;
            } else {
                report_once(
                    &mut quiet,
                    rank,
                    format_args!(
                        "rank {peer} connected again from {addr}; the earlier connection is dropped"
                    ),
                );
            }
        }
        if !busy {
            // Look again in a millisecond while the constructor is still waiting
            // for a rank and the list has room: the connections that form a mesh
            // can land just after a pass, and every stagger that misses one costs
            // a whole `ACCEPT_POLL` — 30 ms of formation on a 16-rank loopback
            // mesh, against 6 ms here. A full list has nothing to take, so a flood
            // cannot turn this into a spin. With no budget (`RLX_CONNECT_WAIT=0`, a
            // persistent node waiting on the next job) it lasts until the rank
            // dials: a wakeup a millisecond, the price of catching a peer that
            // lands between ticks.
            let fast = greetings.waiting() && silent.len() < MAX_SILENT;
            thread::sleep(if fast {
                Duration::from_millis(1)
            } else {
                ACCEPT_POLL
            });
        }
    }
}

/// Report `msg` once per episode of trouble rather than on every tick: the
/// failures here are transient (a full descriptor table, a buffer shortage), and a
/// flood of connections that say nothing — or a stale peer that keeps re-dialling —
/// would otherwise log per connection. `quiet` is cleared when a rank greets for
/// the first time, so once a mesh has formed the first problem reported silences
/// the acceptor for the rest of the transport's life: one line per transport, not
/// one per connection.
fn report_once(quiet: &mut bool, rank: u32, msg: std::fmt::Arguments<'_>) {
    if !*quiet {
        eprintln!("[rlx-driver] rank {rank}: {msg}");
        *quiet = true;
    }
}

/// Is a whole frame header waiting on `stream`? `Ok(None)` while it is still
/// arriving, `Err` when the connection is gone. Any kind is reported: what it is,
/// and whether it may be consumed, is the caller's judgement.
fn header_waiting(stream: &TcpStream) -> io::Result<Option<(u8, u32, u32, u32)>> {
    let mut hdr = [0u8; HEADER_LEN];
    match stream.peek(&mut hdr) {
        Ok(0) => Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
        Ok(n) if n >= HEADER_LEN => Ok(Some(parse_header(&hdr))),
        Ok(_) => Ok(None),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// Full-mesh TCP transport implementing both [`Transport`] (two-sided
/// send/recv) and [`SymmetricTransport`] (one-sided put/get/barrier).
/// Construct via [`TcpTransport::bind`] or [`ThunderboltTransport::bind`].
pub struct NetTransport {
    inner: Arc<NetInner>,
    readers: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// The thread that keeps accepting peers for the transport's life.
    acceptor: Option<JoinHandle<()>>,
}

impl NetTransport {
    /// Build a transport from an already-bound listener. Ranks above us
    /// we connect to; ranks below us we accept from (each connector
    /// announces itself with a `HELLO`). `peers[r]` is rank `r`'s listen
    /// address; `peers[rank]` must match `listener`.
    ///
    /// Returns once every rank we accept from has greeted, or
    /// [`io::ErrorKind::TimedOut`] naming the ones that did not greet within
    /// `RLX_CONNECT_WAIT` — a rank that never arrives is an error, not a block.
    /// No peer can stretch that budget: a connection that has not said `HELLO` yet
    /// is set aside rather than read where it stands. A flood of such connections
    /// can still spend it before a genuine peer is taken (see `accept_loop`).
    ///
    /// A slot converges on the last greeting heard, not on the provably right
    /// connection: a stale greeting can fill it before the genuine peer dials, and
    /// the wire carries no session identity to tell the two apart. So this can
    /// return while a rank's slot still holds the connection the genuine peer will
    /// replace — or that stays, if it never dials — and bytes written to that rank
    /// in that window go to the old socket and are lost. A peer that greets again
    /// later takes the slot from the genuine connection the same way, losing the
    /// bytes in flight to it — both are excluded on `accept_loop`. What holds is
    /// only that a slot is never empty and never holds a connection that did not
    /// greet for that rank.
    ///
    /// The listener stays bound for the transport's life, because a genuine peer
    /// that dials after a stale greeting is still taken; a caller that binds a
    /// fixed port again must drop this transport first.
    ///
    /// Public so callers (and multi-rank tests) can pre-bind ephemeral
    /// ports, exchange the addresses out-of-band, then construct the
    /// mesh — avoiding the fixed-port races [`TcpTransport::bind`] is
    /// subject to.
    pub fn from_listener(
        rank: u32,
        world: u32,
        listener: TcpListener,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<Self> {
        Self::from_listener_reversed(rank, world, listener, peers, heap_size, &[])
    }

    /// [`NetTransport::from_listener`], with the connection of each pair in
    /// `reversed` (`(lower, higher)` ranks) opened by its higher rank.
    pub fn from_listener_reversed(
        rank: u32,
        world: u32,
        listener: TcpListener,
        peers: Vec<SocketAddr>,
        heap_size: usize,
        reversed: &[(u32, u32)],
    ) -> io::Result<Self> {
        assert_eq!(
            peers.len(),
            world as usize,
            "peers must have world_size entries"
        );
        assert!(rank < world, "rank out of range");

        let inner = Arc::new(NetInner {
            rank,
            world,
            heap_size,
            writers: (0..world).map(|_| Mutex::new(None)).collect(),
            inbox: Mutex::new(HashMap::new()),
            inbox_cv: Condvar::new(),
            heap: RwLock::new(vec![0u8; heap_size]),
            getresp: Mutex::new(HashMap::new()),
            getresp_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            lost: (0..world).map(|_| AtomicU64::new(0)).collect(),
            conn_gen: (0..world).map(|_| AtomicU64::new(0)).collect(),
            recv_timeout: env_recv_timeout(),
        });
        let readers: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));

        // Connect to every higher-ranked peer, and to the lower ones of
        // reversed pairs; accept from the rest. A peer's listener is bound
        // before it dials anyone, so the order cannot deadlock.
        let flipped = |a: u32, b: u32| reversed.contains(&(a.min(b), a.max(b)));
        let dial: Vec<u32> = (0..world)
            .filter(|&p| p != rank && ((p > rank) != flipped(rank, p)))
            .collect();
        let expect: BTreeSet<u32> = (0..world)
            .filter(|&p| p != rank && !dial.contains(&p))
            .collect();
        for p in dial {
            let stream = connect_retry(peers[p as usize])?;
            stream.set_nodelay(true).ok();
            let mut wr = stream.try_clone()?;
            write_frame(&mut wr, HELLO, rank, 0, &[])?;
            let rd = stream.try_clone()?;
            let inner2 = inner.clone();
            readers
                .lock()
                .unwrap()
                .push(thread::spawn(move || reader_loop(rd, inner2, p, 0)));
            *inner.writers[p as usize].lock().unwrap() = Some(wr);
        }

        // Accept the rest for as long as the transport lives — a rank's slot can
        // be filled again by a genuine peer that dials after a stale greeting —
        // and wait here until every one has greeted.
        let (acceptor, greetings) = if expect.is_empty() {
            (None, None)
        } else {
            listener.set_nonblocking(true)?;
            let greetings = Arc::new(Greetings::new(expect, env_connect_wait()));
            let (inner2, readers2, greetings2) =
                (inner.clone(), readers.clone(), greetings.clone());
            let acceptor =
                thread::spawn(move || accept_loop(listener, inner2, readers2, greetings2));
            (Some(acceptor), Some(greetings))
        };
        // Built before the wait so a rank that never arrives drops this and
        // takes the links, the readers and the heap with it.
        let transport = Self {
            inner,
            readers,
            acceptor,
        };
        if let Some(greetings) = &greetings {
            // A rank that never arrives is an error naming it, not a block.
            greetings.wait(rank)?;
        }
        Ok(transport)
    }
}

impl NetTransport {
    fn empty_inner(rank: u32, world: u32, heap_size: usize) -> Arc<NetInner> {
        Arc::new(NetInner {
            rank,
            world,
            heap_size,
            writers: (0..world).map(|_| Mutex::new(None)).collect(),
            inbox: Mutex::new(HashMap::new()),
            inbox_cv: Condvar::new(),
            heap: RwLock::new(vec![0u8; heap_size]),
            getresp: Mutex::new(HashMap::new()),
            getresp_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            lost: (0..world).map(|_| AtomicU64::new(0)).collect(),
            conn_gen: (0..world).map(|_| AtomicU64::new(0)).collect(),
            recv_timeout: env_recv_timeout(),
        })
    }

    /// Coordinator side of a **star** topology: bind `listener` (rank 0's
    /// address) and accept `world - 1` worker connections, each announcing its
    /// rank with a `HELLO`. Workers connect only to the coordinator (no
    /// worker↔worker links), so a worker needs **no inbound port** — the
    /// NAT/mobile-friendly shape for iOS, phones, and Pis behind a router.
    pub fn coordinator_listen(
        world: u32,
        listener: TcpListener,
        heap_size: usize,
    ) -> io::Result<Self> {
        let inner = Self::empty_inner(0, world, heap_size);
        let mut readers = Vec::new();
        for _ in 1..world {
            let (stream, _addr) = listener.accept()?;
            stream.set_nodelay(true).ok();
            let mut rd = stream.try_clone()?;
            let (kind, peer, _b, _payload) = read_frame(&mut rd)?;
            if kind != HELLO || peer == 0 || peer >= world {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected HELLO from a worker rank, got kind={kind} rank={peer}"),
                ));
            }
            let wr = stream.try_clone()?;
            let inner2 = inner.clone();
            readers.push(thread::spawn(move || reader_loop(rd, inner2, peer, 0)));
            *inner.writers[peer as usize].lock().unwrap() = Some(wr);
        }
        Ok(Self {
            inner,
            readers: Arc::new(Mutex::new(readers)),
            acceptor: None,
        })
    }

    /// Worker side of a **star** topology: dial the coordinator at `coord_addr`
    /// and announce this `rank`. No listener — the worker only needs outbound
    /// connectivity, so it works from behind NAT / on mobile.
    pub fn worker_dial(
        rank: u32,
        world: u32,
        coord_addr: SocketAddr,
        heap_size: usize,
    ) -> io::Result<Self> {
        assert!(rank > 0 && rank < world, "worker rank must be in 1..world");
        let inner = Self::empty_inner(rank, world, heap_size);
        let stream = connect_retry(coord_addr)?;
        stream.set_nodelay(true).ok();
        let mut wr = stream.try_clone()?;
        write_frame(&mut wr, HELLO, rank, 0, &[])?;
        let rd = stream.try_clone()?;
        let inner2 = inner.clone();
        let readers = vec![thread::spawn(move || reader_loop(rd, inner2, 0, 0))];
        *inner.writers[0].lock().unwrap() = Some(wr);
        Ok(Self {
            inner,
            readers: Arc::new(Mutex::new(readers)),
            acceptor: None,
        })
    }
}

impl Drop for NetTransport {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        // The acceptor stops taking connections first, so no reader is added
        // while the sockets are being closed. It polls `shutdown` and never
        // blocks on a peer, so the join returns within one tick.
        if let Some(thread) = self.acceptor.take() {
            let _ = thread.join();
        }
        // Closing each socket makes the reader threads' read_exact fail.
        for w in &self.inner.writers {
            if let Some(s) = w.lock().unwrap().as_ref() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
        self.inner.inbox_cv.notify_all();
        self.inner.getresp_cv.notify_all();
        if let Ok(mut hs) = self.readers.lock() {
            for h in hs.drain(..) {
                let _ = h.join();
            }
        }
    }
}

impl Transport for NetTransport {
    fn rank(&self) -> u32 {
        self.inner.rank
    }
    fn world_size(&self) -> u32 {
        self.inner.world
    }
    fn send_bytes(&self, to: u32, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError> {
        if to == self.inner.rank {
            // Self-message: deliver straight to our own inbox.
            self.inner
                .inbox
                .lock()
                .unwrap()
                .entry((to, tag))
                .or_default()
                .push_back(bytes.to_vec());
            self.inner.inbox_cv.notify_all();
            return Ok(());
        }
        self.inner.send_frame(to, SEND, self.inner.rank, tag, bytes)
    }
    fn recv_bytes(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
        self.inner.recv_inbox(from, tag)
    }
    fn recv_bytes_timeout(
        &self,
        from: u32,
        tag: u32,
        timeout: Duration,
    ) -> Result<Option<Vec<u8>>, CollectiveError> {
        Ok(self.inner.recv_inbox_timeout(from, tag, timeout))
    }
    fn barrier(&self) -> Result<(), CollectiveError> {
        default_barrier(self)
    }
}

impl SymmetricTransport for NetTransport {
    fn num_ranks(&self) -> u32 {
        self.inner.world
    }
    fn this_rank(&self) -> Rank {
        Rank(self.inner.rank)
    }
    fn put(&self, buf: SymmetricBuffer, src: &[u8]) -> Result<(), CollectiveError> {
        self.inner.check_buf(buf)?;
        if src.len() != buf.len {
            return Err(CollectiveError::LengthMismatch {
                expected: buf.len,
                got: src.len(),
            });
        }
        if buf.rank.0 == self.inner.rank {
            let mut h = self.inner.heap.write().unwrap();
            h[buf.offset..buf.offset + buf.len].copy_from_slice(src);
            Ok(())
        } else {
            self.inner
                .send_frame(buf.rank.0, PUT, self.inner.rank, buf.offset as u32, src)
        }
    }
    fn get(&self, buf: SymmetricBuffer, dst: &mut [u8]) -> Result<(), CollectiveError> {
        self.inner.check_buf(buf)?;
        if dst.len() != buf.len {
            return Err(CollectiveError::LengthMismatch {
                expected: buf.len,
                got: dst.len(),
            });
        }
        if buf.rank.0 == self.inner.rank {
            let h = self.inner.heap.read().unwrap();
            dst.copy_from_slice(&h[buf.offset..buf.offset + buf.len]);
            Ok(())
        } else {
            let mut payload = [0u8; 8];
            payload[0..4].copy_from_slice(&(buf.offset as u32).to_le_bytes());
            payload[4..8].copy_from_slice(&(buf.len as u32).to_le_bytes());
            self.inner
                .send_frame(buf.rank.0, GETREQ, self.inner.rank, 0, &payload)?;
            let data = self.inner.wait_getresp(buf.rank.0);
            if data.len() != buf.len {
                return Err(CollectiveError::LengthMismatch {
                    expected: buf.len,
                    got: data.len(),
                });
            }
            dst.copy_from_slice(&data);
            Ok(())
        }
    }
    fn barrier(&self) -> Result<(), CollectiveError> {
        default_barrier(self)
    }
}

/// Default symmetric-heap size per rank (bytes) when a constructor is
/// not given one. 64 MiB comfortably holds a single transformer hidden
/// state at typical batch×seq×d_model for collective scratch.
pub const DEFAULT_HEAP_BYTES: usize = 64 * 1024 * 1024;

/// Portable TCP transport. Works over any reachable IP — Ethernet or
/// the macOS Thunderbolt Bridge link. This is the baseline that
/// pipeline- and tensor-parallel inference run on before any fast path.
pub struct TcpTransport;

impl TcpTransport {
    /// Bind this rank's listener at `peers[rank]`'s port, on every local
    /// address, and establish the full mesh. The ranks' lists agree on every
    /// port; a peer may name this rank by another of its addresses (an iPad
    /// over its USB-C link rather than Wi-Fi), so the listener cannot be
    /// bound to the one address this rank's own list gives — it was, and a
    /// rank on its USB-C link refused every connection.
    pub fn bind(
        rank: u32,
        world: u32,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<NetTransport> {
        let listener = listen_any(peers[rank as usize])?;
        NetTransport::from_listener_reversed(
            rank,
            world,
            listener,
            peers,
            heap_size,
            &env_mesh_reverse(),
        )
    }
}

/// A listener on `at`'s port on every local address — `[::]`, dual-stack,
/// except on Windows, whose IPv6 sockets take no IPv4 — else at `at` itself.
fn listen_any(at: SocketAddr) -> io::Result<TcpListener> {
    let v4 = SocketAddr::new(
        std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
        at.port(),
    );
    let v6 = SocketAddr::new(
        std::net::IpAddr::V6(std::net::Ipv6Addr::UNSPECIFIED),
        at.port(),
    );
    let first = if cfg!(windows) { v4 } else { v6 };
    TcpListener::bind(first)
        .or_else(|_| TcpListener::bind(v4))
        .or_else(|_| TcpListener::bind(at))
}

/// Thunderbolt transport. Same wire protocol as [`TcpTransport`] today
/// (TCP over the Thunderbolt Bridge IP link), but a distinct type so
/// the symmetric one-sided heap path is the intended interface and so a
/// future zero-copy Thunderbolt DMA backend can replace the engine
/// behind the unchanged [`Transport`] + [`SymmetricTransport`] traits.
pub struct ThunderboltTransport;

impl ThunderboltTransport {
    /// Bind at `peers[rank]`. The addresses are expected to be on the
    /// Thunderbolt interface (see [`ThunderboltTransport::looks_like_thunderbolt`]);
    /// correctness does not depend on it, but bandwidth does.
    pub fn bind(
        rank: u32,
        world: u32,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<NetTransport> {
        TcpTransport::bind(rank, world, peers, heap_size)
    }

    /// Heuristic: is `ip` plausibly a Thunderbolt Bridge address? macOS
    /// auto-assigns link-local `169.254.0.0/16` to the bridge when no
    /// static IP is configured; static setups commonly use a private
    /// `10.0.0.0/8` block. This is advisory only.
    pub fn looks_like_thunderbolt(ip: IpAddr) -> bool {
        match ip {
            IpAddr::V4(v4) => {
                let o = v4.octets();
                o[0] == 169 && o[1] == 254 || o[0] == 10
            }
            IpAddr::V6(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    /// A dial to an address that never answers gives up within its
    /// budget: one blocked `connect` took over a minute a try.
    ///
    /// `backend-tests`: seconds of wall clock, no product path.
    #[cfg(feature = "backend-tests")]
    #[test]
    fn a_dial_that_gets_no_answer_ends_within_its_budget() {
        let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "1");
        let t = std::time::Instant::now();
        // TEST-NET-1 (RFC 5737): routed nowhere.
        let r = super::connect_retry("192.0.2.1:9".parse().unwrap());
        assert!(r.is_err());
        assert!(
            t.elapsed() < std::time::Duration::from_secs(4),
            "took {:?}",
            t.elapsed()
        );
    }

    /// Send this rank's byte to every other rank of a three-rank mesh and
    /// collect theirs, sorted.
    fn exchange(t: &NetTransport, rank: u32) -> Vec<u8> {
        for p in (0..3).filter(|&p| p != rank) {
            t.send_bytes(p, 7, &[rank as u8]).unwrap();
        }
        let mut got: Vec<u8> = (0..3)
            .filter(|&p| p != rank)
            .map(|p| t.recv_bytes(p, 7).unwrap()[0])
            .collect();
        got.sort();
        got
    }

    /// A mesh whose pairs (0,2) and (1,2) are opened by rank 2 carries
    /// messages both ways on every link, like the usual one.
    #[test]
    fn a_reversed_pair_is_dialed_by_its_higher_rank() {
        let listeners: Vec<TcpListener> = (0..3)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let peers: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(r, l)| {
                let peers = peers.clone();
                std::thread::spawn(move || {
                    let t = NetTransport::from_listener_reversed(
                        r as u32,
                        3,
                        l,
                        peers,
                        64,
                        &[(0, 2), (1, 2)],
                    )
                    .unwrap();
                    exchange(&t, r as u32)
                })
            })
            .collect();
        let got: Vec<Vec<u8>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
    }

    /// A stale process's HELLO (a persistent worker from the run before,
    /// still dialing) and a bare port probe, queued at a rank's listener
    /// before the real ranks connect, neither take a real rank's place: the
    /// genuine peer is still accepted, the mesh forms, every pair carries
    /// messages, and the replaced connection closing marks no one lost.
    ///
    /// The interleaving is forced, not hoped for: the accept queue is FIFO, so
    /// the probe and the stale greeting are taken first and rank 0's
    /// connection next, and rank 1 dials only once rank 0's transport is built.
    /// An acceptor that stops at one greeting per rank leaves rank 1's genuine
    /// connection queued, and the exchange below fails.
    ///
    /// Every rank thread scopes its own bounds (`env::scoped` is thread-local):
    /// the mesh must form and the exchange must complete well inside them, so a
    /// regression fails this test instead of blocking the suite.
    #[test]
    fn a_stale_hello_does_not_take_a_ranks_place() {
        let listeners: Vec<TcpListener> = (0..3)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let peers: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        // A probe that says nothing and goes, then an old "rank 1", whose
        // process — like a persistent worker's — is still up.
        drop(TcpStream::connect(peers[2]).unwrap());
        let mut stale = TcpStream::connect(peers[2]).unwrap();
        write_frame(&mut stale, HELLO, 1, 0, &[]).unwrap();
        stale
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();

        let mut it = listeners.into_iter();
        let (l0, l1, l2) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());

        // Rank 0 dials ranks 1 and 2, so its connection to rank 2 is queued
        // behind the stale one before rank 1 exists at all.
        let (built, rank0_built) = std::sync::mpsc::channel();
        let peers0 = peers.clone();
        let rank0 = std::thread::spawn(move || {
            let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "5");
            let _recv = rlx_ir::env::scoped("RLX_COLLECTIVE_TIMEOUT", "10");
            let t = NetTransport::from_listener(0, 3, l0, peers0, 64).unwrap();
            built.send(()).unwrap();
            exchange(&t, 0)
        });
        rank0_built
            .recv_timeout(Duration::from_secs(10))
            .expect("rank 0 never built");

        let peers1 = peers.clone();
        let rank1 = std::thread::spawn(move || {
            let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "5");
            let _recv = rlx_ir::env::scoped("RLX_COLLECTIVE_TIMEOUT", "10");
            let t = NetTransport::from_listener(1, 3, l1, peers1, 64).unwrap();
            let got = exchange(&t, 1);
            // Rank 2 asks only once the genuine connection has replaced the
            // stale one: a rank the replacement marked lost would fail this.
            t.recv_bytes(2, 8).unwrap();
            t.send_bytes(2, 9, &[1]).unwrap();
            got
        });

        let peers2 = peers.clone();
        let (built_tx, built_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        let rank2 = std::thread::spawn(move || {
            let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "5");
            let _recv = rlx_ir::env::scoped("RLX_COLLECTIVE_TIMEOUT", "10");
            let t = NetTransport::from_listener(2, 3, l2, peers2, 64).unwrap();
            built_tx.send(()).unwrap();
            // Send only once the genuine connection has replaced the stale one: a
            // message sent before that is lost with the connection it went to.
            go_rx.recv().unwrap();
            let got = exchange(&t, 2);
            t.send_bytes(1, 8, &[2]).unwrap();
            assert_eq!(t.recv_bytes(1, 9).unwrap(), vec![1]);
            got
        });

        built_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("rank 2 never built");
        // The acceptor closes the connection it replaced, and that close is the
        // only signal a caller has that the slot now holds the genuine peer: it is
        // published after the slot was filled.
        let mut buf = [0u8; 1];
        match stale.read(&mut buf) {
            Ok(0) => {}
            Ok(n) => panic!("the stale connection carried {n} bytes"),
            Err(e) => panic!("the stale greeting was never replaced: {e}"),
        }
        go_tx.send(()).unwrap();

        let got = vec![
            rank0.join().unwrap(),
            rank1.join().unwrap(),
            rank2.join().unwrap(),
        ];
        assert_eq!(got, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
    }

    /// A rank that never dials is an error that names it, within the budget a
    /// dial gets.
    ///
    /// `backend-tests`: seconds of wall clock, no product path.
    #[cfg(feature = "backend-tests")]
    #[test]
    fn a_rank_that_never_greets_is_an_error_that_names_it() {
        let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "1");
        let listeners: Vec<TcpListener> = (0..2)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let peers: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        // Rank 1 accepts from rank 0, which never dials.
        let l1 = listeners.into_iter().nth(1).unwrap();
        let t = Instant::now();
        let err = NetTransport::from_listener(1, 2, l1, peers, 64)
            .err()
            .expect("a peer that never greets must fail the mesh");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            err.to_string().contains("rank 0"),
            "error should name the rank: {err}"
        );
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "took {:?}",
            t.elapsed()
        );
    }

    /// A connection that says nothing — a port probe, a half-open socket — sits
    /// ahead of the real peer in the accept queue. It must not be read where it
    /// stands: the peer behind it would wait out that read while the
    /// constructor's budget ran, and a healthy mesh would fail. Run on its own
    /// thread with a receive timeout, so a version that blocks on the probe fails
    /// here instead of holding the suite.
    #[test]
    fn a_silent_connection_does_not_time_the_mesh_out() {
        let listeners: Vec<TcpListener> = (0..2)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let peers: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let mut it = listeners.into_iter();
        let (l0, l1) = (it.next().unwrap(), it.next().unwrap());
        // The probe says nothing and stays, first in rank 1's queue.
        let probe = TcpStream::connect(peers[1]).unwrap();

        let peers0 = peers.clone();
        let rank0 = std::thread::spawn(move || {
            let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "5");
            let t = NetTransport::from_listener(0, 2, l0, peers0, 64).unwrap();
            t.send_bytes(1, 7, &[0]).unwrap();
            t
        });

        let rank1 = std::thread::spawn(move || {
            let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "2");
            // Captured by the transport, so the receive below is bounded too.
            let _recv = rlx_ir::env::scoped("RLX_COLLECTIVE_TIMEOUT", "10");
            NetTransport::from_listener(1, 2, l1, peers, 64)
        });
        // Waited for, not blocked on: a version that reads the probe where it
        // stands holds that thread, and the suite must not wait with it.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !rank1.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            rank1.is_finished(),
            "the silent connection was read where it stood"
        );
        let t1 = rank1
            .join()
            .expect("rank 1 panicked")
            .expect("a silent connection ahead of the peer must not time the mesh out");
        assert_eq!(t1.recv_bytes(0, 7).unwrap(), vec![0]);
        drop(t1);
        drop(probe);
        drop(rank0.join().unwrap());
    }

    /// A connection that says nothing is never dropped to make room: while the
    /// accept path's list is full it pauses and the kernel backlog holds the
    /// rest, because the connection it would drop could be a genuine peer, and a
    /// genuine peer never re-dials.
    ///
    /// `backend-tests`: seconds of wall clock, no product path.
    #[cfg(feature = "backend-tests")]
    #[test]
    fn a_full_silent_list_does_not_drop_a_connection() {
        let l1 = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = l1.local_addr().unwrap();
        let probes: Vec<TcpStream> = (0..MAX_SILENT + 1)
            .map(|_| TcpStream::connect(addr).unwrap())
            .collect();
        let peers = vec!["127.0.0.1:1".parse::<SocketAddr>().unwrap(), addr];
        let waiting = thread::spawn(move || {
            let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "2");
            NetTransport::from_listener(1, 2, l1, peers, 64)
                .err()
                .map(|e| e.kind())
        });
        // Longer than a tick, so the list is full and the last probe is queued.
        thread::sleep(Duration::from_millis(500));
        for (i, p) in probes.iter().enumerate() {
            p.set_nonblocking(true).unwrap();
            let mut b = [0u8; 1];
            match p.peek(&mut b) {
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Ok(0) => panic!("probe {i} was dropped to make room"),
                Ok(_) => panic!("probe {i} carried a byte"),
                Err(e) => panic!("probe {i}: {e}"),
            }
        }
        assert_eq!(waiting.join().unwrap(), Some(io::ErrorKind::TimedOut));
    }

    /// The bound is absolute: connections that say nothing, or that drip their
    /// `HELLO` a byte at a time, cannot postpone it — they are set aside and
    /// dropped, and the wait still ends on the budget.
    ///
    /// `backend-tests`: seconds of wall clock, no product path.
    #[cfg(feature = "backend-tests")]
    #[test]
    fn a_probe_flood_cannot_postpone_the_timeout() {
        let l1 = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = l1.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (flood_stop, drip_stop) = (stop.clone(), stop.clone());
        let flood = thread::spawn(move || {
            let mut held = Vec::new();
            while !flood_stop.load(Ordering::SeqCst) {
                if let Ok(s) = TcpStream::connect(addr) {
                    held.push(s);
                }
                thread::sleep(Duration::from_millis(200));
            }
            held
        });
        let drip = thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            while !drip_stop.load(Ordering::SeqCst) {
                if s.write_all(&[1u8]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(300));
            }
        });

        let _wait = rlx_ir::env::scoped("RLX_CONNECT_WAIT", "1");
        let peers = vec!["127.0.0.1:1".parse::<SocketAddr>().unwrap(), addr];
        let t = Instant::now();
        let err = NetTransport::from_listener(1, 2, l1, peers, 64)
            .err()
            .expect("rank 0 never dials");
        let took = t.elapsed();
        stop.store(true, Ordering::SeqCst);
        drop(flood.join().unwrap());
        drip.join().unwrap();

        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            err.to_string().contains("rank 0"),
            "error should name the rank: {err}"
        );
        assert!(took < Duration::from_secs(3), "the bound slid: {took:?}");
    }

    use super::*;
    use crate::collective::{ReduceKind, all_reduce};
    use crate::transport::ProcessGroup;

    /// Pre-bind `n` loopback listeners and run `body(rank, transport)`
    /// on a thread per rank. Returns once every rank's body returns.
    /// A peer that exits mid-job fails the receive waiting on it (with its
    /// rank) instead of hanging; data it sent before exiting still arrives.
    #[test]
    fn receive_from_a_dead_peer_errors_instead_of_hanging() {
        let listeners: Vec<TcpListener> = (0..2)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let mut it = listeners.into_iter();
        let (l0, l1) = (it.next().unwrap(), it.next().unwrap());
        let a1 = addrs.clone();
        let peer = thread::spawn(move || {
            let t = NetTransport::from_listener(1, 2, l1, a1, 1024).unwrap();
            t.send_bytes(0, 7, b"last words").unwrap();
            // `t` drops here: the process "dies".
        });
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let t = NetTransport::from_listener(0, 2, l0, addrs, 1024).unwrap();
            let first = t.recv_bytes(1, 7);
            let second = t.recv_bytes(1, 7);
            tx.send((first, second)).unwrap();
        });
        peer.join().unwrap();
        let (first, second) = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("receive on a dead peer hung");
        assert_eq!(first.unwrap(), b"last words");
        let err = second.expect_err("dead peer must error").to_string();
        assert!(err.contains("rank 1"), "error should name the rank: {err}");
    }

    fn run_net<F>(n: u32, heap: usize, body: F)
    where
        F: Fn(u32, Arc<NetTransport>) + Send + Sync + 'static,
    {
        let listeners: Vec<TcpListener> = (0..n)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let body = Arc::new(body);
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let body = body.clone();
                thread::spawn(move || {
                    let t = Arc::new(
                        NetTransport::from_listener(rank as u32, n, listener, addrs, heap)
                            .expect("transport build"),
                    );
                    body(rank as u32, t.clone());
                    // Trailing barrier: keep every rank's reader threads alive
                    // until all peers finish, so a late remote GET still gets a
                    // response before any transport is dropped.
                    <NetTransport as Transport>::barrier(&*t).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Star variant of [`run_net`]: rank 0 listens (coordinator) and ranks
    /// 1..n dial in (workers, no inbound port of their own).
    fn run_star<F>(n: u32, heap: usize, body: F)
    where
        F: Fn(u32, Arc<NetTransport>) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let coord_addr = listener.local_addr().unwrap();
        let body = Arc::new(body);
        let mut handles = Vec::new();
        {
            let body = body.clone();
            handles.push(thread::spawn(move || {
                let t = Arc::new(NetTransport::coordinator_listen(n, listener, heap).unwrap());
                body(0, t.clone());
                <NetTransport as Transport>::barrier(&*t).unwrap();
            }));
        }
        for rank in 1..n {
            let body = body.clone();
            handles.push(thread::spawn(move || {
                let t = Arc::new(NetTransport::worker_dial(rank, n, coord_addr, heap).unwrap());
                body(rank, t.clone());
                <NetTransport as Transport>::barrier(&*t).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn star_dial_out_coordinator_worker() {
        // Workers dial the coordinator (no listener of their own); the
        // coordinator sums their contributions and sends the result back.
        run_star(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            if rank == 0 {
                let a = g.recv_f32(1, 5).unwrap();
                let b = g.recv_f32(2, 5).unwrap();
                let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                g.send_f32(1, 6, &sum).unwrap();
                g.send_f32(2, 6, &sum).unwrap();
            } else {
                g.send_f32(0, 5, &[rank as f32]).unwrap();
                assert_eq!(g.recv_f32(0, 6).unwrap(), vec![3.0]); // 1 + 2
            }
        });
    }

    #[test]
    fn two_sided_pipeline_handoff() {
        // rank r sends [r*10..] to r-1 (toward the leader), like a
        // pipeline returning hidden states.
        run_net(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            let n = g.world_size();
            if rank + 1 < n {
                let got = g.recv_f32(rank + 1, 7).unwrap();
                assert_eq!(got, vec![(rank as f32 + 1.0) * 10.0, 1.0]);
            }
            if rank > 0 {
                g.send_f32(rank - 1, 7, &[rank as f32 * 10.0, 1.0]).unwrap();
            }
        });
    }

    #[test]
    fn process_group_all_reduce_over_tcp() {
        run_net(4, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            let mut data = vec![rank as f32 + 1.0; 5];
            g.all_reduce(&mut data, ReduceKind::Sum).unwrap();
            assert_eq!(data, vec![10.0; 5], "rank {rank}");
        });
    }

    #[test]
    fn federated_average_skips_late_worker() {
        // rank 2 behaves like an offline/slow edge client — it only sends
        // after the deadline, so rank 0 drops it and averages ranks 0 and 1.
        run_net(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            let mut data = vec![rank as f32 + 1.0; 3]; // 1, 2, 3
            if rank == 2 {
                std::thread::sleep(Duration::from_millis(300));
            }
            let present = g
                .federated_average(&mut data, Duration::from_millis(100))
                .unwrap();
            if rank == 0 {
                assert_eq!(present, 2, "late rank 2 must be dropped");
                assert_eq!(data, vec![1.5; 3]); // mean of ranks 0 and 1
            }
        });
    }

    #[test]
    fn process_group_barrier_and_broadcast() {
        run_net(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            g.barrier().unwrap();
            let mut data = if rank == 0 {
                vec![1.0, 2.0]
            } else {
                vec![0.0, 0.0]
            };
            g.broadcast(0, &mut data).unwrap();
            assert_eq!(data, vec![1.0, 2.0], "rank {rank}");
        });
    }

    #[test]
    fn symmetric_remote_put_get() {
        // Rank 0 PUTs into rank 1's heap; rank 1 reads its own slot and
        // sees it. Then rank 1 GETs from rank 0's heap.
        run_net(2, 4096, |rank, t| {
            let off = 128;
            let payload = [5u8, 6, 7, 8];
            if rank == 0 {
                // seed our own heap for rank 1 to GET later
                t.put(
                    SymmetricBuffer {
                        rank: Rank(0),
                        offset: off,
                        len: 4,
                    },
                    &[1u8, 2, 3, 4],
                )
                .unwrap();
                // write into rank 1's heap
                t.put(
                    SymmetricBuffer {
                        rank: Rank(1),
                        offset: off,
                        len: 4,
                    },
                    &payload,
                )
                .unwrap();
            }
            // Make sure the PUT landed before the reads.
            <NetTransport as Transport>::barrier(&t).unwrap();
            // ^ disambiguates Transport::barrier from SymmetricTransport::barrier
            if rank == 1 {
                let mut got = [0u8; 4];
                t.get(
                    SymmetricBuffer {
                        rank: Rank(1),
                        offset: off,
                        len: 4,
                    },
                    &mut got,
                )
                .unwrap();
                assert_eq!(got, payload, "local read of remote PUT");
                let mut remote = [0u8; 4];
                t.get(
                    SymmetricBuffer {
                        rank: Rank(0),
                        offset: off,
                        len: 4,
                    },
                    &mut remote,
                )
                .unwrap();
                assert_eq!(remote, [1u8, 2, 3, 4], "remote GET from rank 0");
            }
        });
    }

    #[test]
    fn symmetric_collective_all_reduce_over_net() {
        // The existing one-sided collective (collective::all_reduce)
        // runs unmodified over NetTransport.
        run_net(4, 4096, |rank, t| {
            let elems = 3usize;
            let bytes = elems * 4;
            let buf = SymmetricBuffer {
                rank: Rank(rank),
                offset: 0,
                len: bytes,
            };
            let mut local = vec![rank as f32 + 1.0; elems];
            all_reduce(&*t, buf, &mut local, ReduceKind::Sum).unwrap();
            assert_eq!(local, vec![10.0; elems], "rank {rank}");
        });
    }

    #[test]
    fn thunderbolt_addr_heuristic() {
        use std::net::Ipv4Addr;
        assert!(ThunderboltTransport::looks_like_thunderbolt(
            Ipv4Addr::new(169, 254, 3, 1).into()
        ));
        assert!(ThunderboltTransport::looks_like_thunderbolt(
            Ipv4Addr::new(10, 0, 0, 2).into()
        ));
        assert!(!ThunderboltTransport::looks_like_thunderbolt(
            Ipv4Addr::new(192, 168, 1, 5).into()
        ));
    }
}
