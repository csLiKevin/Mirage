const packager = require("electron-packager");

const whitelist = [
    "",
    "/package.json",
    "/application",
    "/application/index.html",
    "/application/main.js"
];

packager({
    asar: true,
    dir: ".",
    ignore: path => !whitelist.includes(path),
    name: "Mirage",
    out: "build/client",
    overwrite: true
});
