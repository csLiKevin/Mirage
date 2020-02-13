<script context="module" type="text/javascript">
    if (window.__dl8__tracker === undefined) {
        const script = document.createElement("script");
        script.type = "text/javascript";
        script.src =
            "../static/cdn.delight-vr.com/1.6.8/dl8-0c65c31ebaf3b71dd6b6a452405e9011ff75d027.js";
        document.head.appendChild(script);
    }
</script>

<script type="text/javascript">
    export let filePath = null;

    $: {
        // Play the new video when filePath changes.
        filePath;
        const player = document.querySelector("dl8-video");
        if (player !== null) {
            player.exit();
            player.start();
        }
    }
</script>

<style type="text/css">
    .vr-player {
        max-height: 100vh;
    }

    /* Hide the player's close button. */
    .vr-player :global(#dl8-content-container .dl8Icon.iconCross) {
        display: none;
    }
</style>

<dl8-video class="vr-player" format="STEREO_180_LR">
    <source src={filePath} type="video/mp4" />
</dl8-video>
