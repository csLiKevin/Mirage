import videojs, { VideoJsPlayer } from "video.js";
import "videojs-vr";

import "video.js/dist/video-js.min.css";
// Prevent videojs-vr from changing the BigPlayButton's styles.
// import "videojs-vr/dist/videojs-vr.css";

interface VRVideoJsPlayer extends VideoJsPlayer {
    camera: {
        position: {
            x: number;
            y: number;
            z: number;
        };
    };
    vr: (options?: { [key: string]: string }) => VRVideoJsPlayer;
}

const player = videojs("video") as VRVideoJsPlayer;
player.vr({ projection: "180_LR" });

// Prevent videos from starting automatically if the play button was clicked before the video was loaded.
player.addClass("vjs-show-big-play-button-on-pause");
player.on("loadedmetadata", player.pause);

const containerElement = document.getElementById("container");

containerElement.addEventListener("dragover", (event) => {
    event.preventDefault();
});

containerElement.addEventListener("drop", (event) => {
    event.preventDefault();
    const {
        dataTransfer: { files },
    } = event;

    if (!files.length) {
        return;
    }

    const file = files[0];
    player.src({ type: file.type, src: URL.createObjectURL(file) });
});

document.addEventListener("keydown", (event) => {
    const { code } = event;

    if (code === "Space") {
        event.preventDefault();
        player.paused() ? player.play() : player.pause();
        return;
    }

    if (code === "KeyR") {
        event.preventDefault();
        const {
            camera: { position },
        } = player.vr();
        position.x = 0;
        position.y = 0;
        position.z = 0;
        return;
    }
});
