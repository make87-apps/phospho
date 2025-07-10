import logging
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import uvicorn
from make87_messages.core.header_pb2 import Header
from make87_messages.text.text_plain_pb2 import PlainText
from phosphobot.am import Pi0
from phosphobot.app import app
from fastapi.middleware.cors import CORSMiddleware
import make87
from phosphobot.camera import AllCameras

logger = logging.getLogger(__name__)

PHOSPHO_SERVER_PORT = 8473

async def run_model():
    config = make87.config.load_config_from_env()
    pi_interface = config.interfaces.get("pi0-interface", None)
    if pi_interface is None:
        logger.info(
            "No pi0-interface found in the configuration. "
            "Please ensure that the interface is correctly configured."
        )
        return
    pi_client = pi_interface.clients.get("pi0", None)
    if pi_client is None:
        logger.info(
            "No pi0 client found in the configuration. "
            "Please ensure that the client is correctly configured."
        )
        return

    PHOSPHOBOT_API_URL = f"http://localhost:{PHOSPHO_SERVER_PORT}"

    # Get a camera frame
    allcameras = AllCameras()

    # Need to wait for the cameras to initialize
    time.sleep(1)

    # Instantiate the model
    model = Pi0(server_url=pi_client.vpn_ip, server_port=pi_client.vpn_port)

    while True:
        # Get the frames from the cameras
        # We will use this model: PLB/pi0-so100-orangelegobrick-wristcam
        # It requires 2 cameras (a context cam and a wrist cam)
        images = [
            allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
            allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
        ]

        # Get the robot state
        state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()

        inputs = {
            "state": np.array(state["angles_rad"]),
            "images": np.array(images),
            "prompt": "Pick up the screw driver",
        }

        # Go through the model
        actions = model(inputs)

        for action in actions:
            # Send the new joint postion to the robot
            httpx.post(
                f"{PHOSPHOBOT_API_URL}/joints/write", json={"angles": action.tolist()}
            )
            # Wait to respect frequency control (30 Hz)
            time.sleep(1 / 30)



async def run_app():
    app.user_middleware = [
        m for m in app.user_middleware
        if m.cls is not CORSMiddleware
    ]
    app.middleware_stack = app.build_middleware_stack()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(app=app, host="0.0.0.0", port=PHOSPHO_SERVER_PORT, reload=False)
    server = uvicorn.Server(config)
    await server.serve()



async def main():
    app_task = asyncio.create_task(run_app())
    model_task = asyncio.create_task(run_model())

    await asyncio.gather(app_task, model_task)


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down the server...")
