import resolve from "rollup-plugin-node-resolve";
import svelte from "rollup-plugin-svelte";
import commonjs from "rollup-plugin-commonjs";

export default {
    input: ["src/app.js"],
    output: {
        dir: "static",
        format: "cjs"
    },
    plugins: [resolve({
        browser: true,
        dedupe: importee => importee === 'svelte' || importee.startsWith('svelte/')
    }), commonjs(), svelte()],
};
