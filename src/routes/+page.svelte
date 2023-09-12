<script lang="ts">
    import { onMount } from "svelte";
    import type { VideoJsPlayer } from "video.js";
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

    let player: VRVideoJsPlayer;

    function handleDrop(event: DragEvent) {
        const { dataTransfer } = event;
        const { files } = dataTransfer!;

        if (!files.length) {
            return;
        }

        const file = files[0];
        player.src({ type: file.type, src: URL.createObjectURL(file) });
    }

    function handleKeyDown(event: KeyboardEvent) {
        const { code } = event;

        if (code === "KeyR") {
            event.preventDefault();
            const {
                camera: { position },
            } = player.vr();
            position.x = 0;
            position.y = 0;
            position.z = 0;
        }

        if (code === "Space") {
            event.preventDefault();
            player.paused() ? player.play() : player.pause();
        }
    }

    onMount(async () => {
        const videojs = await import("video.js");
        await import("videojs-vr");

        player = videojs.default("video") as VRVideoJsPlayer;
        player.vr({ projection: "180_LR" });

        // Prevent videos from starting automatically if the play button was clicked before the video was loaded.
        player.addClass("vjs-show-big-play-button-on-pause");
        player.on("loadedmetadata", player.pause);
    });
</script>

<svelte:body
    on:dragover|preventDefault
    on:drop|preventDefault={handleDrop}
    on:keydown={handleKeyDown}
/>
<div id="container">
    <!-- svelte-ignore a11y-media-has-caption -->
    <video id="video" class="video-js" controls preload="auto" />
</div>

<style global>
    body {
        margin: 0;
    }

    #container {
        height: 100vh;
        width: 100vw;
    }

    #container > .video-js {
        height: inherit;
        width: inherit;
    }

    #container > .video-js > .vjs-big-play-button {
        left: 50%;
        top: 50%;
        margin-left: calc(-3em / 2);
        margin-top: calc(-1.5em / 2);
    }
</style>
