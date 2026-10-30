const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const path = require('path')
const fs = require('fs')

const ROOT = path.join(__dirname, '..')
const DATA_ROOT = path.join(ROOT, 'appdata')
const ALLOWED_DIRS = new Set(['saves', 'experiments', 'exports', 'logs'])
const ALLOWED_EXT = new Set(['.json', '.csv'])

function dirPath(dir) {
  if (!ALLOWED_DIRS.has(dir)) return null
  return path.join(DATA_ROOT, dir)
}

function safeJoin(dir, name) {
  const base = dirPath(dir)
  if (!base) return null
  const clean = String(name || '').replace(/[\\/:*?"<>|\0]/g, '_').trim()
  if (!clean) return null
  return path.join(base, clean)
}

function ensureDirs() {
  for (const d of ALLOWED_DIRS) {
    const p = dirPath(d)
    if (p) fs.mkdirSync(p, { recursive: true })
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1500,
    height: 940,
    minWidth: 1000,
    minHeight: 640,
    backgroundColor: '#0a0d11',
    title: 'Artificial Life',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      spellcheck: false
    }
  })
  win.removeMenu()
  win.loadFile(path.join(ROOT, 'dist', 'index.html'))
  if (process.env.ALC_DEVTOOLS === '1') win.webContents.openDevTools()
  return win
}

ipcMain.handle('data:list', (_e, dir) => {
  ensureDirs()
  const base = dirPath(dir)
  if (!base) throw new Error('invalid directory')
  return fs.readdirSync(base).filter((f) => {
    try {
      return fs.statSync(path.join(base, f)).isFile()
    } catch {
      return false
    }
  })
})

ipcMain.handle('data:read', (_e, dir, name) => {
  const p = safeJoin(dir, name)
  if (!p) throw new Error('invalid path')
  try {
    return fs.readFileSync(p, 'utf8')
  } catch {
    return null
  }
})

ipcMain.handle('data:write', (_e, dir, name, content) => {
  const p = safeJoin(dir, name)
  if (!p) throw new Error('invalid path')
  ensureDirs()
  const tmp = p + '.tmp'
  fs.writeFileSync(tmp, content, 'utf8')
  fs.renameSync(tmp, p)
  return true
})

ipcMain.handle('data:remove', (_e, dir, name) => {
  const p = safeJoin(dir, name)
  if (!p) return false
  try {
    fs.unlinkSync(p)
    return true
  } catch {
    return false
  }
})

ipcMain.handle('data:stats', (_e, dir, name) => {
  const p = safeJoin(dir, name)
  if (!p) throw new Error('invalid path')
  try {
    return fs.statSync(p).mtimeMs
  } catch {
    return null
  }
})

ipcMain.handle('data:resolve', (_e, dir) => {
  const base = dirPath(dir)
  return base || null
})

ipcMain.handle('app:error-dialog', async (_e, title, message) => {
  await dialog.showMessageBox(createWindow(), {
    type: 'error',
    title: String(title || 'Error'),
    message: String(message || '')
  })
})

app.whenReady().then(() => {
  ensureDirs()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  app.quit()
})
