import commonjs from "@rollup/plugin-commonjs";
import postcss from "rollup-plugin-postcss";
import resolve from "@rollup/plugin-node-resolve";

export default {
    input: "index.ts",
    output: {
        file: "index.js",
    },
    plugins: [
        commonjs(),
        resolve({
            browser: true,
        }),
        postcss(),
    ],
};
