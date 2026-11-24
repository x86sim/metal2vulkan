//! x86Sim: MoltenVK's MTLEvent-based Vulkan semaphores, as x86Sim's mtlvk replays them: every
//! event a shared event, one queue, and per frame three command buffers committed in order:
//!   acquire: signal A = n                    (vkAcquireNextImageKHR)
//!   submit:  wait A >= n, work, signal R = m (vkQueueSubmit)
//!   present: wait R >= m, signal A = n       (vkQueuePresentKHR: re-signals a value A reached)
//! with A rotating over three events and R over two, and the host waiting for the submit two
//! frames back (vkWaitForFences). Every wait is for a value an earlier commit of the same queue
//! signals, so nothing may stall; under the iOS simulator one did on the old runtime, after 25 to
//! 215 frames.
//!
//! The second test adds what x86Sim's metalhostd does besides: another thread reads every event's
//! value (`signaled_value`) while the frames run, as Apple's simulator server does to answer
//! listeners.
//!
//! Device test: compiled only under `m2v_gpu` / `m2v_cpu_icd`, on every device of the cfg's kind.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::{CommandBuffer, CommandBufferStatus, MemoryClass};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const FRAMES: u64 = 3000;
    const STALL: Duration = Duration::from_secs(5);

    #[test]
    fn moltenvk_event_semaphores_never_stall() {
        each_device("moltenvk_semaphores", |name, _, device| {
            frames(name, &device, false)
        });
    }

    #[test]
    fn moltenvk_event_semaphores_never_stall_while_values_are_read() {
        each_device("moltenvk_semaphores_read", |name, _, device| {
            frames(name, &device, true)
        });
    }

    /// Whether `command_buffer` completed within [`STALL`].
    fn completes(command_buffer: &CommandBuffer) -> bool {
        let deadline = Instant::now() + STALL;
        loop {
            match command_buffer.status() {
                CommandBufferStatus::Completed => return true,
                CommandBufferStatus::Error => return false,
                _ if Instant::now() >= deadline => return false,
                _ => std::thread::sleep(Duration::from_micros(200)),
            }
        }
    }

    fn frames(name: &str, device: &crate::Device, read_values: bool) {
        let queue = device.new_command_queue(64).expect("a queue");
        let block = device
            .take_block(MemoryClass::Shared, 4096)
            .expect("a block");
        let target = device
            .new_buffer(block.id, 0, 4096, Some(0))
            .expect("a buffer");
        let acquire: Vec<_> = (0..3)
            .map(|_| device.new_shared_event().expect("an event"))
            .collect();
        let render: Vec<_> = (0..2)
            .map(|_| device.new_shared_event().expect("an event"))
            .collect();
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

            let cb = queue.command_buffer().expect("a command buffer");
            cb.encode_signal_event(acquire[a].as_event(), n)
                .expect("a signal");
            cb.commit().expect("a commit");

            let cb = queue.command_buffer().expect("a command buffer");
            cb.encode_wait_for_event(acquire[a].as_event(), n)
                .expect("a wait");
            let mut blit = cb.blit_command_encoder().expect("a blit encoder");
            blit.fill_buffer(&target, 0..256, (frame & 0xff) as u8)
                .expect("a fill");
            blit.end_encoding();
            cb.encode_signal_event(render[r].as_event(), m)
                .expect("a signal");
            cb.commit().expect("a commit");
            submits.push(cb);

            let cb = queue.command_buffer().expect("a command buffer");
            cb.encode_wait_for_event(render[r].as_event(), m)
                .expect("a wait");
            cb.encode_signal_event(acquire[a].as_event(), n)
                .expect("a signal");
            cb.commit().expect("a commit");

            if submits.len() > 2 {
                let old = submits.remove(0);
                assert!(
                    completes(&old),
                    "{name}: frame {frame}: the submit of frame {} did not complete in {STALL:?} \
                     (status {:?}); acquire {acquired:?} render {rendered:?}, values A {:?} R {:?}",
                    frame - 2,
                    old.status(),
                    acquire
                        .iter()
                        .map(|e| e.signaled_value())
                        .collect::<Vec<_>>(),
                    render
                        .iter()
                        .map(|e| e.signaled_value())
                        .collect::<Vec<_>>(),
                );
            }
        }
        for cb in submits {
            assert!(completes(&cb), "{name}: the last submits complete");
        }
        stop.store(true, Ordering::Relaxed);
        if let Some(reader) = reader {
            reader.join().expect("the reader thread");
        }
        let bytes = target.contents().expect("a Shared buffer's contents");
        // SAFETY: the buffer's first 256 bytes, which no command writes any more.
        let first = unsafe { *bytes.as_ptr() };
        assert_eq!(
            first,
            ((FRAMES - 1) & 0xff) as u8,
            "{name}: the last frame's fill"
        );
    }
}
