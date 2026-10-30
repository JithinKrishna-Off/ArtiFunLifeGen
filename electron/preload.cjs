const { contextBridge, ipcRenderer } = require('electron')

const invoke = (chan) => (...args) => ipcRenderer.invoke(chan, ...args)

contextBridge.exposeInMainWorld('dataBridge', {
  list: invoke('data:list'),
  read: invoke('data:read'),
  write: invoke('data:write'),
  remove: invoke('data:remove'),
  stats: invoke('data:stats'),
  resolve: invoke('data:resolve'),
  errorDialog: invoke('app:error-dialog')
})
