from __future__ import annotations

import argparse

import numpy as np

from colosseum_policy_server import ColosseumPolicySDK, ImageFrame, Observation
from colosseum_policy_server.demo_actions import make_gripper_action_chunk


def _print_observation(observation: Observation) -> None:
    print(
        f"\nsequence={observation.sequence} control_step={observation.control_step} "
        f"deadline_ms={observation.deadline_ms}"
    )
    print(f"instruction={observation.instruction!r}")
    print(f"state.joints={observation.state.joints}")
    print(f"state.gripper={observation.state.gripper}")
    for name in ("left_image", "right_image", "head_image"):
        image = getattr(observation.state, name)
        print(f"state.{name}={None if image is None else image.shape}")


class OpenCVViewer:
    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("OpenCV is required. Run: uv sync --extra demo") from exc
        self.cv2 = cv2

    def show(self, frame: ImageFrame) -> None:
        cv2 = self.cv2
        if frame.encoding in {"JPEG", "PNG"}:
            image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode {frame.image_id} as {frame.encoding}")
        elif frame.encoding == "RAW_RGB":
            expected = frame.width * frame.height * 3
            if len(frame.data) != expected:
                raise ValueError(
                    f"{frame.image_id} RAW_RGB has {len(frame.data)} bytes; expected {expected}"
                )
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"unsupported image encoding: {frame.encoding}")
        cv2.imshow(f"Colosseum - {frame.image_id}", image)

    def should_quit(self) -> bool:
        return self.cv2.waitKey(1) & 0xFF in {27, ord("q")}

    def close(self) -> None:
        self.cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print Colosseum observations, display images, and optionally send "
            "fake gripper actions."
        )
    )
    parser.add_argument(
        "--enable-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send the demo gripper action chunk (enabled by default)",
    )
    args = parser.parse_args()

    viewer = OpenCVViewer()
    active_session = ""
    gripper_target = 1.0
    if not args.enable_action:
        print(
            "Observation-only mode: no action will be sent. "
            "Remove --no-enable-action to send the demo chunk."
        )
    else:
        print("Action mode enabled: demo gripper action chunks will be sent.")

    try:
        with ColosseumPolicySDK.from_yaml("configs/policy.yaml") as sdk:
            print(f"Connected to Router at {sdk.router_url} as policy server {sdk.server_id}.")
            print("Waiting for a Robot Client and observation data...")
            while True:
                observation = sdk.get_obs()
                _print_observation(observation)
                for image in observation.images:
                    viewer.show(image)

                if args.enable_action:
                    if sdk.robot is None:
                        raise RuntimeError("Router did not provide the Robot Client configuration")
                    if observation.session_id != active_session:
                        active_session = observation.session_id
                        gripper_target = 1.0
                    hold_steps = min(sdk.max_horizon, sdk.robot.control_hz)
                    chunk = make_gripper_action_chunk(
                        observation,
                        sdk.robot,
                        sdk.selected_action_space,
                        (gripper_target,) * hold_steps,
                    )
                    print(
                        f"sending action chunk shape={chunk.shape} "
                        f"gripper_target={gripper_target:.0f} "
                        f"duration={hold_steps / sdk.robot.control_hz:.2f}s"
                    )
                    sdk.send_action(chunk)
                    gripper_target = 1.0 - gripper_target

                if viewer.should_quit():
                    break
                print("Waiting for the next observation...")
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
