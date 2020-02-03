const { app, BrowserWindow, session } = require("electron");
const { join } = require("path");
const staticPath = join(__dirname, "..", "static");

let browserWindow;

function createWindow() {
    browserWindow = new BrowserWindow({
        frame: false,
        height: 720,
        width: 1280
    });
    // browserWindow.removeMenu();
    browserWindow.loadFile("src/index.html");
    browserWindow.on("closed", () => {
        // Dereference the window object, usually you would store windows
        // in an array if your app supports multi windows, this is the time
        // when you should delete the corresponding element.
        browserWindow = null;
    });

    // Reroute all external urls to the static folder.
    session.defaultSession.webRequest.onBeforeRequest(
        { urls: ["file://*"] },
        (details, callback) => {
            const url = new URL(details.url);
            if (url.hostname) {
                callback({
                    redirectURL: join(staticPath, url.hostname, url.pathname)
                });
            } else {
                callback({});
            }
        }
    );
}

app.on("ready", createWindow);

app.on("window-all-closed", () => {
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
