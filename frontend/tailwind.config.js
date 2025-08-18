// tailwind.config.js

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // <-- This tells Tailwind to scan all your component files
  ],
  darkMode: 'class', 
  safelist: [
    'from-purple-400',
    'to-pink-400',
    'bg-purple-50',
    'from-blue-400',
    'to-cyan-400',
    'bg-blue-50',
    'from-green-400',
    'to-teal-400',
    'bg-green-50',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2B3A67',
        secondary: '#00B8D4',
        accent: '#FF9900',
        'light-bg': '#F8F9FA',
        'dark-panel': '#1E293B',
      },
      fontFamily: {
        // Add this line
        'dancing': ['"Dancing Script"', 'cursive'],
      },
    },
  },
  plugins: [],
}