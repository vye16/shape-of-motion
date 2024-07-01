import threading
import time

import viser


def add_gui_playback_group(
    server: viser.ViserServer,
    num_frames: int,
    min_fps: float = 1.0,
    max_fps: float = 60.0,
    fps_step: float = 0.1,
    initial_fps: float = 10.0,
):
    gui_timestep = server.gui.add_slider(
        "Timestep",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
        disabled=True,
    )
    gui_next_frame = server.gui.add_button("Next Frame")
    gui_prev_frame = server.gui.add_button("Prev Frame")
    gui_playing_pause = server.gui.add_button("Pause")
    gui_playing_pause.visible = False
    gui_playing_resume = server.gui.add_button("Resume")
    gui_framerate = server.gui.add_slider(
        "FPS", min=min_fps, max=max_fps, step=fps_step, initial_value=initial_fps
    )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    def _toggle_gui_playing(_):
        gui_playing_pause.visible = not gui_playing_pause.visible
        gui_playing_resume.visible = not gui_playing_resume.visible
        gui_timestep.disabled = gui_playing_pause.visible
        gui_next_frame.disabled = gui_playing_pause.visible
        gui_prev_frame.disabled = gui_playing_pause.visible

    gui_playing_pause.on_click(_toggle_gui_playing)
    gui_playing_resume.on_click(_toggle_gui_playing)

    # Create a thread to update the timestep indefinitely.
    def _update_timestep():
        while True:
            if gui_playing_pause.visible:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1 / gui_framerate.value)

    threading.Thread(target=_update_timestep, daemon=True).start()

    return (
        gui_timestep,
        gui_next_frame,
        gui_prev_frame,
        gui_playing_pause,
        gui_playing_resume,
        gui_framerate,
    )
