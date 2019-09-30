const { app, BrowserWindow, session } = require("electron");
const express = require("express");

let browserWindow;
let server;

function createWindow() {
    browserWindow = new BrowserWindow({ height: 600, width: 800 });

    browserWindow.loadFile("application/index.html");
    browserWindow.openDevTools();
    browserWindow.on("closed", () => {
        // Dereference the window object, usually you would store windows
        // in an array if your app supports multi windows, this is the time
        // when you should delete the corresponding element.
        browserWindow = null;
    });

    server = express()
        .use(
            express.static("static", {
                setHeaders: (response, path) => {
                    response.setHeader("Access-Control-Allow-Origin", "null");
                }
            })
        )
        .listen(3000);

    // Reroute all external requests.
    session.defaultSession.webRequest.onBeforeRequest(
        { urls: ["*://*/*"] },
        (details, callback) => {
            const url = new URL(details.url);
            if (url.hostname !== "localhost") {
                callback({
                    redirectURL: `http://localhost:3000/${url.hostname}${url.pathname}${url.search}`
                });
            } else {
                callback({});
            }
        }
    );
}

app.on("ready", createWindow);

app.on("window-all-closed", () => {
    server.close();

    // Quit the app when all windows are closed except on macOS; on macOS
    // applications and their menu bar stays active until the user quits
    // explicitly.
    if (process.platform !== "darwin") {
        app.quit();
    }
});

app.on("activate", () => {
    // On macOS re-create a window in the app when the dock icon is clicked and
    // there are no other windows open.
    if (browserWindow === null) {
        createWindow();
    }
});
