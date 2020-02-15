<script context="module" type="text/javascript">
    import FileDropArea from "./FileDropArea.svelte";
    import Player from "./Player.svelte";
</script>

<FileDropArea let:filePath>
    <Player {filePath} />
</FileDropArea>
