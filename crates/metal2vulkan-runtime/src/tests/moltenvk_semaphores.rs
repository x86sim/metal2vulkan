//! MoltenVK's MTLEvent-based Vulkan semaphores, as x86Sim's mtlvk replays them: every event a shared event,
//! one queue, and per frame three command buffers committed in order:
//!   acquire: signal A = n                  (vkAcquireNextImageKHR)
//!   submit:  wait A >= n, work, signal R = m (vkQueueSubmit)
//!   present: wait R >= m, signal A = n       (vkQueuePresentKHR: re-signals a value A already reached)
//! with A rotating over three events and R over two, and the host waiting for the submit two frames back
//! (vkWaitForFences). Every wait is for a value an earlier commit of the same queue signals, so nothing may
//! stall; under the iOS simulator one did, after 25 to 215 frames.
//!
//! The second test adds what x86Sim's metalhostd does besides: another thread reads every event's value
//! (`signaled_value`) while the frames run, as Apple's simulator server does to answer listeners.
//!
//! Device test: compiled only under `m2v_gpu` / `m2v_cpu_icd`.

use super::blit_holds::{device, Scratch};
use crate::{CommandBuffer, EventRef, ResourceOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const FRAMES: u64 = 3000;
const STALL: Duration = Duration::from_secs(5);

#[test]
fn moltenvk_event_semaphores_never_stall() {
    frames(false);
}

#[test]
fn moltenvk_event_semaphores_never_stall_while_values_are_read() {
    frames(true);
}

fn frames(read_values: bool) {
    let scratch = Scratch::new(if read_values {
        "mvk-sem-read"
    } else {
        "mvk-sem"
    });
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let acquire: Vec<_> = (0..3).map(|_| device.new_shared_event().unwrap()).collect();
    let render: Vec<_> = (0..2).map(|_| device.new_shared_event().unwrap()).collect();
    let stop = Arc::new(AtomicBool::new(false));
    let reader = read_values.then(|| {
        let (events, stop) = (
            acquire.iter().chain(&render).cloned().collect::<Vec<_>>(),
            Arc::clone(&stop),
        );
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                for event in &events {
                    event.signaled_value();
                }
            }
        })
    });
    let mut acquired = [0u64; 3];
    let mut rendered = [0u64; 2];
    let mut submits: Vec<CommandBuffer> = Vec::new();
    for frame in 0..FRAMES {
        let (a, r) = ((frame % 3) as usize, (frame % 2) as usize);
        acquired[a] += 1;
        rendered[r] += 1;
        let (n, m) = (acquired[a], rendered[r]);

        let cb = queue.command_buffer().unwrap();
        cb.encode_signal_event(EventRef::Shared(&acquire[a]), n)
            .unwrap();
        cb.commit().unwrap();

        let cb = queue.command_buffer().unwrap();
        cb.encode_wait_for_event(EventRef::Shared(&acquire[a]), n)
            .unwrap();
        let mut blit = cb.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..256, (frame & 0xff) as u8)
            .unwrap();
        blit.end_encoding().unwrap();
        cb.encode_signal_event(EventRef::Shared(&render[r]), m)
            .unwrap();
        cb.commit().unwrap();
        submits.push(cb);

        let cb = queue.command_buffer().unwrap();
        cb.encode_wait_for_event(EventRef::Shared(&render[r]), m)
            .unwrap();
        cb.encode_signal_event(EventRef::Shared(&acquire[a]), n)
            .unwrap();
        cb.commit().unwrap();

        if submits.len() > 2 {
            let old = submits.remove(0);
            if old.wait_until_completed_for(STALL).is_err() {
                panic!(
                    "frame {frame}: the submit of frame {} did not complete in {STALL:?} (status {:?}); \
                     acquire {acquired:?} render {rendered:?}, values A {:?} R {:?}",
                    frame - 2,
                    old.status(),
                    acquire.iter().map(|e| e.signaled_value()).collect::<Vec<_>>(),
                    render.iter().map(|e| e.signaled_value()).collect::<Vec<_>>(),
                );
            }
        }
    }
    for cb in submits {
        cb.wait_until_completed_for(STALL)
            .expect("the last submits complete");
    }
    stop.store(true, Ordering::Relaxed);
    if let Some(reader) = reader {
        reader.join().unwrap();
    }
}
