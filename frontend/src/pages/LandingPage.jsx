import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Brain, Headphones, ArrowRight, Github, Layers
} from 'lucide-react';
import logo from '../assets/logo.png'; // Make sure this file exists
import { useTheme } from '../context/ThemeContext';
import ThemeToggle from '../components/ui/ThemeToggle';

// --- Vanta.js Background Component (Refined Logic) ---
const VantaBackground = () => {
  const { theme } = useTheme();
  const [vantaEffect, setVantaEffect] = useState(null);
  const vantaRef = useRef(null);

  useEffect(() => {
    // Dynamically load scripts if not already present
    const loadScript = (src) => new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) return resolve();
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.body.appendChild(script);
    });

    // Initialize Vanta
    loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js')
      .then(() => loadScript('https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.net.min.js'))
      .then(() => {
        if (window.VANTA && vantaRef.current) {
          const effect = window.VANTA.NET({
            el: vantaRef.current,
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.0,
            minWidth: 200.0,
            scale: 1.0,
            scaleMobile: 1.0,
            color: 0x6b21a8,
            backgroundColor: theme === 'dark' ? 0x0 : 0xfafafa,
            points: 12.0,
            maxDistance: 25.0,
            spacing: 18.0,
          });
          setVantaEffect(effect);
        }
      });

    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
    // This effect should only run once on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // This effect ONLY runs when the theme changes to update the background color
  useEffect(() => {
    if (vantaEffect) {
      vantaEffect.setOptions({
        backgroundColor: theme === 'dark' ? 0x0 : 0xfafafa,
      });
    }
  }, [vantaEffect, theme]);

  return <div ref={vantaRef} className="fixed inset-0 w-full h-full pointer-events-none z-0" />;
};


// --- Custom Hook for Intersection Observer ---
const useIntersectionObserver = (options) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsIntersecting(true);
        if(ref.current) observer.unobserve(ref.current);
      }
    }, options);
    if (ref.current) {
      observer.observe(ref.current);
    }
    return () => {
      if (ref.current) {
        // eslint-disable-next-line react-hooks/exhaustive-deps
        observer.unobserve(ref.current);
      }
    };
  }, [options]);
  return [ref, isIntersecting];
};


// --- Main Landing Page Component ---
const AnimatedLandingPage = () => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [reduceMotion, setReduceMotion] = useState(false);

  const [heroRef, heroIsVisible] = useIntersectionObserver({ threshold: 0.1 });
  const [featuresRef, featuresAreVisible] = useIntersectionObserver({ threshold: 0.1 });

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduceMotion(mediaQuery.matches);
    const queryListener = () => setReduceMotion(mediaQuery.matches);
    mediaQuery.addEventListener('change', queryListener);
    return () => mediaQuery.removeEventListener('change', queryListener);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => setCurrentFeature(prev => (prev + 1) % 3), 4000);
    return () => clearInterval(interval);
  }, []);

  const features = [/* ... */];
  const smoothScrollTo = (ref) => ref.current?.scrollIntoView({ behavior: 'smooth' });

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: `
          @keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
          @keyframes glow { 0% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.2); } 100% { box-shadow: 0 0 40px rgba(139, 92, 246, 0.4); } }
          @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
      ` }} />
      
      <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-black dark:text-gray-100 overflow-x-hidden transition-colors duration-500">
        {!reduceMotion && <VantaBackground />}
        
        <header className="relative z-50 bg-white/80 dark:bg-black/80 backdrop-blur-xl border-b border-gray-200 dark:border-white/10 sticky top-0">
          <div className="container mx-auto px-6 py-4">
            <div className="flex justify-between items-center">
              <div 
                className="group cursor-pointer flex items-center space-x-3" 
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              >
                <img 
                  src={logo}
                  alt="weavedocs icon" 
                  className="h-10 w-auto" 
                />
                <span className="font-dancing text-4xl text-gray-900 dark:text-white group-hover:opacity-80 transition-opacity duration-300">
                  weavedocs
                </span>
              </div>
              
              <nav className="hidden md:flex items-center space-x-6">
                <button onClick={() => smoothScrollTo(featuresRef)} className="text-gray-600 dark:text-white/70 hover:text-gray-900 dark:hover:text-white transition-colors duration-300 font-medium">Features</button>
                <a href="https://github.com/your-repo-link" target="_blank" rel="noopener noreferrer" className="flex items-center text-gray-600 dark:text-white/70 hover:text-gray-900 dark:hover:text-white transition-colors duration-300 font-medium">
                  <Github className="w-4 h-4 mr-2" /> GitHub
                </a>
                <ThemeToggle />
              </nav>
            </div>
          </div>
        </header>

        <section ref={heroRef} className="relative min-h-screen flex items-center justify-center overflow-hidden">
          <div className="container mx-auto px-6 py-24 text-center relative z-10">
            <div className={`transition-all duration-1000 ease-out ${heroIsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
              <h1 className="text-6xl md:text-8xl font-black mb-8 leading-tight">
                <div style={!reduceMotion ? { animation: 'slideUp 1s ease-out 0.2s both' } : {}}>Connect Your Documents.</div>
                <div className="bg-gradient-to-r from-purple-600 via-blue-500 to-cyan-500 bg-clip-text text-transparent" style={!reduceMotion ? { animation: 'gradient 3s ease infinite, slideUp 1s ease-out 0.4s both', backgroundSize: '200% 200%' } : {}}>Uncover Insights.</div>
              </h1>
              
              <p className="text-xl md:text-2xl text-gray-700 dark:text-white/80 mb-12 max-w-4xl mx-auto leading-relaxed" style={!reduceMotion ? { animation: 'slideUp 1s ease-out 0.6s both' } : {}}>
                An intelligent document reader built for comprehensive analysis. Explore contextual connections and generate powerful insights with our advanced AI engine.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-6 justify-center items-center" style={!reduceMotion ? { animation: 'slideUp 1s ease-out 0.8s both' } : {}}>
                {/* Launch App now points to /reader route */}
                <Link to="/reader" className="group relative inline-flex items-center px-10 py-5 bg-gradient-to-r from-purple-500 via-blue-500 to-cyan-500 text-white font-black rounded-2xl text-lg transition-all duration-500 hover:scale-110 transform shadow-2xl shadow-purple-500/25 overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-blue-600 to-cyan-600 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  <span className="relative flex items-center">
                    Launch App
                    <ArrowRight className="ml-3 w-6 h-6 group-hover:translate-x-2 transition-transform duration-500" />
                  </span>
                </Link>
              </div>
            </div>
          </div>
        </section>
        
        {/* Remember to apply dark: classes to the rest of your page sections! */}
      </div>
    </>
  );
};

export default AnimatedLandingPage;
