const packager = require("electron-packager");

const whitelist = [
    "",
    "/package.json",
    "/src",
    "/src/index.html",
    "/src/main.js"
];

packager({
    asar: true,
    dir: ".",
    ignore: path => !whitelist.includes(path) && !path.startsWith("/static"),
    name: "Mirage",
    out: "build",
    overwrite: true
});
