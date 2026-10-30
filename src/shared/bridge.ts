/**
 * Bridge to the Electron main process for project-local application data
 * (saves/, experiments/, exports/, logs/ under appdata/).
 * Falls back to null when running in a plain browser (npm run dev); analytics
 * persistence must handle that via an in-memory fallback.
 */
export interface DataBridge {
  list(dir: 'saves' | 'experiments' | 'exports' | 'logs'): Promise<string[]>
  read(dir: string, name: string): Promise<string | null>
  write(dir: string, name: string, content: string): Promise<boolean>
  remove(dir: string, name: string): Promise<boolean>
  stats(dir: string, name: string): Promise<number | null>
  resolve(dir: string): Promise<string | null>
  errorDialog(title: string, message: string): Promise<void>
}

declare global {
  interface Window {
    dataBridge?: DataBridge
  }
}

export function bridge(): DataBridge | null {
  return typeof window !== 'undefined' && window.dataBridge ? window.dataBridge : null
}
