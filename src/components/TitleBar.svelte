<script type="text/javascript">
    import remote from "pad-left";
    console.log(remote);
</script>

<style type="text/css">
    button {
        -webkit-app-region: no-drag;
    }

    .titlebar {
        -webkit-app-region: drag;
        background-color: white;
        position: fixed;
        width: 100%;
        z-index: 1000;
    }
</style>

<div class="titlebar">
    <button on:click|once={e => window.close()}>Hello World</button>
</div>
