<style type="text/css">
    button {
        -webkit-app-region: no-drag;
    }

    .title-bar {
        -webkit-app-region: drag;
        background-color: white;
        position: fixed;
        width: 100%;
        z-index: 1000;
    }
</style>

<div class="title-bar">
    <button on:click|once={e => window.close()}>Hello World</button>
</div>
