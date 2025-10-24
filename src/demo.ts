import './style.css'
import { setupCounter } from './counter.ts'

export function renderDemo() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
    <div class="min-h-screen flex items-center justify-center bg-gray-900 text-white">
      <div class="text-center">
        <h1 class="text-5xl font-bold mb-8">Counter App</h1>
        <div class="p-8">
          <button 
            id="counter" 
            type="button"
            class="bg-gray-800 hover:bg-gray-700 border border-transparent hover:border-blue-500 px-6 py-3 rounded-lg font-medium transition-colors duration-200 focus:outline-none focus:ring-4 focus:ring-blue-500/50"
          ></button>
        </div>
      </div>
    </div>
  `

  setupCounter(document.querySelector<HTMLButtonElement>('#counter')!)
}
