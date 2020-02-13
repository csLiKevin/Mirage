<script context="module" type="text/javascript">
    import FileDropArea from "./FileDropArea.svelte";
    import Player from "./Player.svelte";
    import TitleBar from "./TitleBar.svelte";
</script>

<FileDropArea let:filePath>
    <TitleBar />
    <Player {filePath} />
</FileDropArea>
