const { app, BrowserWindow } = require("electron");

let browserWindow;

function createWindow() {
    browserWindow = new BrowserWindow({
        height: 600,
        width: 800,
        webPreferences: { nodeIntegration: true }
    });

    browserWindow.loadFile("application/index.html");
    browserWindow.openDevTools();
    browserWindow.on("closed", () => {
        // Dereference the window object, usually you would store windows
        // in an array if your app supports multi windows, this is the time
        // when you should delete the corresponding element.
        browserWindow = null;
    });
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
