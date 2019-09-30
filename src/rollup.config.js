import resolve from "rollup-plugin-node-resolve";
import svelte from "rollup-plugin-svelte";

export default {
    input: ["src/vr_player.js"],
    output: {
        dir: "static/js",
        format: "cjs"
    },
    plugins: [resolve(), svelte()]
};
