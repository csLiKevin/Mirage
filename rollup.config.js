import resolve from "rollup-plugin-node-resolve";
import svelte from "rollup-plugin-svelte";

export default {
    input: ["src/player.js"],
    output: {
        dir: "static",
        format: "cjs"
    },
    plugins: [resolve(), svelte()]
};
