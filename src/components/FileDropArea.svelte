<script type="text/javascript">
    let filePath = null;

    function handleDrop(event) {
        const {
            dataTransfer: { files }
        } = event;
        filePath = files.length ? files[0].path : null;
    }
</script>

<div on:dragover|preventDefault on:drop|preventDefault={handleDrop}>
    <slot {filePath} />
</div>
