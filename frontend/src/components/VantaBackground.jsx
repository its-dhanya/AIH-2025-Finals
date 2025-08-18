import React, { useState, useEffect, useRef } from 'react';
import { useTheme } from '../context/ThemeContext'; // Assumes you have this context

const VantaBackground = () => {
  const { theme } = useTheme();
  const [vantaEffect, setVantaEffect] = useState(null);
  const vantaRef = useRef(null);

  useEffect(() => {
    // Dynamically load scripts
    const loadScript = (src) => new Promise((resolve) => {
      if (document.querySelector(`script[src="${src}"]`)) return resolve();
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = resolve;
      document.body.appendChild(script);
    });

    // Initialize Vanta
    loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js')
      .then(() => loadScript('https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.waves.min.js'))
      .then(() => {
        if (window.VANTA && !vantaEffect) {
          setVantaEffect(window.VANTA.WAVES({ el: vantaRef.current }));
        }
      });

    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
  }, [vantaEffect]);

  // Update Vanta colors when the theme changes
  useEffect(() => {
    if (vantaEffect) {
      vantaEffect.setOptions({
        color: theme === 'dark' ? 0x5052c : 0x7079b,
        waveHeight: 15,
        shininess: 50,
        waveSpeed: 0.75,
        zoom: 0.85,
      });
    }
  }, [vantaEffect, theme]);

  return <div ref={vantaRef} className="fixed inset-0 w-full h-full z-0" />;
};

export default VantaBackground;