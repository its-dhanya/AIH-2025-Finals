// PDFViewer.jsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useTheme } from '../../context/ThemeContext';

const PDFViewer = React.memo(({ filePromise, fileName, onViewerReady, onTextSelect, pageNumber, onDocumentLoad, highlightTarget, onOutlineReady }) => {
  const viewerRef = useRef(null);
  const adobeViewRef = useRef(null);
  const adobeViewerRef = useRef(null);
  const enhancedAPIsRef = useRef(null);
  const docIdRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [viewerLoaded, setViewerLoaded] = useState(false);
  const { theme } = useTheme();

  // Replace with your Adobe client ID if different
  const buildId = import.meta.env.VITE_ADOBE_EMBED_API_KEY;

// runtime fallback (recommended if Adobe will run your frontend container)
  const runtimeEnv = (typeof window !== 'undefined' && window.__env) || {};
  const ADOBE_CLIENT_ID = runtimeEnv.VITE_ADOBE_EMBED_API_KEY || buildId

  /* --------------------------------------------------------------------------
     Basic utilities
     -------------------------------------------------------------------------- */

  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  /* --------------------------------------------------------------------------
     Page element finder (robust)
     -------------------------------------------------------------------------- */

  const selectorsForPage = useCallback((p) => ([
    `[data-page-number="${p}"]`,
    `[data-page="${p}"]`,
    `[data-page-index="${p - 1}"]`,
    `.page[data-page-number="${p}"]`,
    `.pdf-page[data-page="${p}"]`,
    `.adobe-page-${p}`,
    `#page-${p}`,
    `.page:nth-child(${p})`,
    `.pdfViewer .page[data-page-number="${p}"]`,
    `.PDFPage[data-page-number="${p}"]`,
    `[aria-label="Page ${p}"]`,
    `[aria-label*="Page ${p}"]`,
    `canvas[data-page-number="${p}"]`,
    `canvas[data-page="${p}"]`,
    `.pageCanvas[data-page-number="${p}"]`,
    `.pageLayer[data-page-number="${p}"]`
  ]), []);

  const findPageElementInDoc = (doc, candidateIndexes) => {
    for (const candidate of candidateIndexes) {
      for (const sel of selectorsForPage(candidate)) {
        try {
          const el = doc.querySelector(sel);
          if (el) return { el, sel, candidate };
        } catch (e) {
          // invalid selector or cross-origin fail
        }
      }

      // Additional dataset numeric check
      try {
        const all = Array.from(doc.querySelectorAll('*'));
        for (const el of all) {
          try {
            const ds = el.dataset;
            if (!ds) continue;
            const keys = Object.keys(ds);
            for (const k of keys) {
              const v = ds[k];
              if (!v) continue;
              if (String(v) === String(candidate) || String(v) === String(candidate - 1)) {
                return { el, sel: `[data-${k}="${v}"]`, candidate };
              }
            }
          } catch (inner) {}
        }
      } catch (e) {}
    }
    return null;
  };

  const waitForPageElement = useCallback(async (targetPage, timeout = 8000, interval = 150) => {
    const container = viewerRef.current;
    if (!container) return null;
    const tNum = Number(targetPage);
    const candidateIndices = !Number.isNaN(tNum) ? [tNum, tNum - 1, tNum + 1, tNum - 2, tNum + 2] : [targetPage];

    const deadline = Date.now() + timeout;
    let found = null;

    while (Date.now() < deadline) {
      // 1) search inside the viewer container
      try {
        found = findPageElementInDoc(container, candidateIndices);
        if (found) {
          console.log('waitForPageElement: found inside viewer container with selector', found.sel);
          return { pageElement: found.el, selector: found.sel, candidate: found.candidate, iframeEl: null };
        }
      } catch (e) {}

      // 2) global document search
      try {
        found = findPageElementInDoc(document, candidateIndices);
        if (found) {
          console.log('waitForPageElement: found in document with selector', found.sel);
          return { pageElement: found.el, selector: found.sel, candidate: found.candidate, iframeEl: null };
        }
      } catch (e) {}

      // 3) Try raw viewer container (if we have adobeViewerRef/enhancedAPIs)
      try {
        const rawViewer = enhancedAPIsRef.current?.getRawViewer?.() || adobeViewerRef.current || null;
        if (rawViewer) {
          const possibleContainers = [
            rawViewer.container,
            rawViewer.viewerContainer,
            rawViewer._container,
            rawViewer.element,
            rawViewer.getViewerElement && rawViewer.getViewerElement(),
            rawViewer.viewerElement
          ].filter(Boolean);

          for (const c of possibleContainers) {
            try {
              const r = findPageElementInDoc(c, candidateIndices);
              if (r && r.el) {
                console.log('waitForPageElement: found inside rawViewer container', r.sel);
                return { pageElement: r.el, selector: r.sel, candidate: r.candidate, iframeEl: null };
              }
            } catch (err) {
              // ignore
            }
          }
        }
      } catch (e) {
        // ignore
      }

      // 4) Search inside same-origin iframes
      try {
        const iframes = document.querySelectorAll('iframe');
        for (const iframe of iframes) {
          try {
            const doc = iframe.contentDocument || iframe.contentWindow?.document;
            if (!doc) continue;
            const r = findPageElementInDoc(doc, candidateIndices);
            if (r && r.el) {
              console.log('waitForPageElement: found inside iframe using selector', r.sel);
              return { pageElement: r.el, selector: r.sel, candidate: r.candidate, iframeEl: iframe };
            }

            // try canvases inside iframe
            const canv = Array.from(doc.querySelectorAll('canvas'));
            for (const c of canv) {
              try {
                const dataPage = c.getAttribute('data-page-number') || c.getAttribute('data-page') || c.dataset?.pageNumber || c.dataset?.page;
                if (dataPage && (String(dataPage) === String(targetPage) || String(dataPage) === String(targetPage - 1))) {
                  return { pageElement: c, selector: 'canvas', candidate: targetPage, iframeEl: iframe };
                }
              } catch (e) {}
            }
          } catch (e) {
            // cross-origin or other access error — ignore
          }
        }
      } catch (e) {}

      // 5) heuristic: try to find elements with aria-label "Page X" across document
      try {
        const aria = document.querySelector(`[aria-label="Page ${targetPage}"], [aria-label*="Page ${targetPage}"]`);
        if (aria) {
          return { pageElement: aria, selector: '[aria-label]', candidate: targetPage, iframeEl: null };
        }
      } catch (e) {}

      await sleep(interval);
    }

    return null;
  }, [findPageElementInDoc, selectorsForPage]);

  /* --------------------------------------------------------------------------
     Text-selection wiring and DOM fallbacks (keep)
     -------------------------------------------------------------------------- */

  const setupTextSelection = useCallback(async (adobeDCView, adobeViewer) => {
    if (!onTextSelect) return;

    console.log('Setting up text selection...');

    let pollInterval = null;

    const handleTextSelection = (event) => {
      console.log('🔍 Processing text selection event:', event.type);
      console.log('🔍 Full event data:', event.data);

      let selectedText = '';

      if (event.type === 'PREVIEW_SELECTION_END' && event.data) {
        const data = event.data;

        for (let i = 0; i <= 10; i++) {
          const pageKey = `page${i}`;
          if (data[pageKey]) {
            const pageData = data[pageKey];

            if (pageData.text) selectedText += pageData.text + ' ';
            if (pageData.selectedText) selectedText += pageData.selectedText + ' ';
            if (pageData.content) selectedText += pageData.content + ' ';

            if (pageData.selections && Array.isArray(pageData.selections)) {
              pageData.selections.forEach(sel => {
                if (sel.text) selectedText += sel.text + ' ';
                if (sel.content) selectedText += sel.content + ' ';
              });
            }

            if (pageData.textRuns && Array.isArray(pageData.textRuns)) {
              pageData.textRuns.forEach(run => {
                if (run.text) selectedText += run.text + ' ';
              });
            }

            const bboxKeys = Object.keys(pageData).filter(key => key.startsWith('bbox'));
            if (bboxKeys.length > 0) {
              bboxKeys.forEach(bboxKey => {
                const bbox = pageData[bboxKey];
                if (bbox.text) selectedText += bbox.text + ' ';
                if (bbox.content) selectedText += bbox.content + ' ';
                if (bbox.textContent) selectedText += bbox.textContent + ' ';

                if (bbox.textData) selectedText += bbox.textData + ' ';
                if (bbox.chars && Array.isArray(bbox.chars)) {
                  bbox.chars.forEach(char => {
                    if (typeof char === 'string') selectedText += char;
                    else if (char.char) selectedText += char.char;
                    else if (char.text) selectedText += char.text;
                  });
                }

                ['words', 'characters', 'spans'].forEach(prop => {
                  if (bbox[prop] && Array.isArray(bbox[prop])) {
                    bbox[prop].forEach(item => {
                      if (typeof item === 'string') selectedText += item + ' ';
                      else if (item.text) selectedText += item.text + ' ';
                      else if (item.char) selectedText += item.char;
                    });
                  }
                });
              });
            }

            ['words', 'lines', 'paragraphs', 'fragments', 'spans', 'characters'].forEach(prop => {
              if (pageData[prop] && Array.isArray(pageData[prop])) {
                pageData[prop].forEach(item => {
                  if (item.text) selectedText += item.text + ' ';
                  if (typeof item === 'string') selectedText += item + ' ';
                });
              }
            });
          }
        }

        ['text', 'selectedText', 'content', 'textContent'].forEach(prop => {
          if (data[prop]) selectedText += data[prop] + ' ';
        });

        if (!selectedText.trim() && data.newSelection) {
          // Try Adobe APIs first, then DOM fallbacks
          const tryAdobeAPI = async () => {
            try {
              if (adobeViewer) {
                const apis = await adobeViewer.getAPIs().catch(() => null);

                if (apis) {
                  const apiMethods = [
                    'getSelectedContent',
                    'getSelectedText',
                    'getSelection',
                    'getCurrentSelection',
                    'extractText',
                    'getTextSelection',
                    'copySelectedText'
                  ];

                  for (const methodName of apiMethods) {
                    if (apis && typeof apis[methodName] === 'function') {
                      try {
                        const result = await apis[methodName]();
                        const text = typeof result === 'string' ? result : result?.text || result?.data || result?.content || result?.selectedText;
                        if (text && text.trim() && text.trim().length > 2) {
                          onTextSelect(text.trim());
                          return true;
                        }
                      } catch (methodError) {
                        // ignore
                      }
                    } else if (typeof adobeViewer[methodName] === 'function') {
                      try {
                        const result = await adobeViewer[methodName]();
                        const text = typeof result === 'string' ? result : result?.text || result?.data || result?.content || result?.selectedText;
                        if (text && text.trim() && text.trim().length > 2) {
                          onTextSelect(text.trim());
                          return true;
                        }
                      } catch (methodError) {
                        // ignore
                      }
                    }
                  }
                }

                const directMethods = ['getSelectedContent', 'getSelectedText', 'copyText'];
                for (const methodName of directMethods) {
                  if (typeof adobeViewer[methodName] === 'function') {
                    try {
                      const result = await adobeViewer[methodName]();
                      const text = typeof result === 'string' ? result : result?.text || result?.data || result?.content;
                      if (text && text.trim() && text.trim().length > 2) {
                        onTextSelect(text.trim());
                        return true;
                      }
                    } catch (methodError) {
                      // ignore
                    }
                  }
                }
              }
            } catch (error) {
              // ignore
            }
            return false;
          };

          const tryDOMSelection = () => {
            let attempts = 0;
            const maxAttempts = 8;

            const attemptSelection = () => {
              attempts++;
              const selectionMethods = [
                () => window.getSelection()?.toString()?.trim(),
                () => document.getSelection()?.toString()?.trim(),
                () => {
                  const iframes = document.querySelectorAll('iframe');
                  for (const iframe of iframes) {
                    try {
                      const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
                      if (iframeDoc) {
                        const sel = iframeDoc.getSelection()?.toString()?.trim();
                        if (sel) return sel;
                      }
                    } catch (e) {}
                  }
                  return '';
                },
                () => {
                  const container = viewerRef.current;
                  if (!container) return '';
                  const selectors = [
                    '[aria-selected="true"]',
                    '.selected',
                    '.highlight',
                    '[data-selected="true"]',
                    '.text-layer .selected',
                    '.adobedc-selection',
                    '.pdf-text-selection',
                    '[style*="background"]',
                    '.textLayer span[style*="background"]'
                  ];
                  for (const selector of selectors) {
                    const elements = container.querySelectorAll(selector);
                    if (elements.length > 0) {
                      const text = Array.from(elements)
                        .map(el => el.textContent || el.innerText)
                        .filter(t => t && t.trim())
                        .join(' ')
                        .trim();
                      if (text) return text;
                    }
                  }
                  return '';
                },
                () => {
                  try {
                    const selection = window.getSelection();
                    if (selection && selection.rangeCount > 0) {
                      const range = selection.getRangeAt(0);
                      const text = range.toString().trim();
                      if (text) return text;
                    }
                  } catch (e) {}
                  return '';
                }
              ];

              for (let methodIndex = 0; methodIndex < selectionMethods.length; methodIndex++) {
                try {
                  const result = selectionMethods[methodIndex]();
                  if (result && result.length > 2) {
                    onTextSelect(result);
                    return true;
                  }
                } catch (error) {}
              }

              if (attempts < maxAttempts) {
                setTimeout(attemptSelection, 100);
              }
            };

            attemptSelection();
          };

          tryAdobeAPI().then(success => {
            if (!success) setTimeout(tryDOMSelection, 10);
          }).catch(() => setTimeout(tryDOMSelection, 10));

          return;
        }
      }

      if (!selectedText) {
        const extractionMethods = [
          () => event.data?.selectedText,
          () => event.data?.text,
          () => event.data?.content,
          () => event.data?.textContent,
          () => typeof event.data === 'string' ? event.data : '',
          () => event.selectedText,
          () => event.text,
          () => event.content
        ];

        for (const method of extractionMethods) {
          try {
            const result = method();
            if (result && typeof result === 'string' && result.trim()) {
              selectedText = result.trim();
              break;
            }
          } catch (error) {}
        }
      }

      selectedText = selectedText.trim();

      if (selectedText && selectedText.length > 2 && selectedText.length < 10000) {
        onTextSelect(selectedText);
      } else {
        setTimeout(() => {
          const selectionMethods = [
            () => window.getSelection()?.toString()?.trim(),
            () => document.getSelection()?.toString()?.trim(),
            () => {
              const container = viewerRef.current;
              if (!container) return '';
              const selectedElements = container.querySelectorAll(
                '[aria-selected="true"], .selected, .highlight, [data-selected="true"], .text-layer .selected'
              );

              if (selectedElements.length > 0) {
                return Array.from(selectedElements)
                  .map(el => el.textContent || el.innerText)
                  .join(' ')
                  .trim();
              }
              return '';
            }
          ];

          for (const method of selectionMethods) {
            try {
              const result = method();
              if (result && result.length > 2) {
                onTextSelect(result);
                return;
              }
            } catch (error) {}
          }
        }, 100);
      }
    };

    const registerAdobeCallback = () => {
      try {
        adobeDCView.registerCallback(
          window.AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
          (event) => {
            const selectionEvents = [
              'TEXT_SELECT', 'TEXT_SELECTED', 'DOCUMENT_FRAGMENT_SELECTED',
              'SELECTION_CHANGE', 'PREVIEW_SELECTION_END', 'PREVIEW_SELECTION_START',
              'ANNOTATION_SELECTED', 'TEXT_COPY', 'COPY_TEXT'
            ];

            if (selectionEvents.includes(event.type)) {
              handleTextSelection(event);
            }

            if (event.type === 'PREVIEW_SELECTION_START' || event.type.includes('MOUSE') || event.type.includes('HOVER')) {
              const immediateSelection = window.getSelection()?.toString()?.trim();
              if (immediateSelection && immediateSelection.length > 2) {
                window.lastCapturedSelection = immediateSelection;
              }
            }
          },
          {
            enableFilePreviewEvents: true,
            enablePDFAnalytics: false,
          }
        );
        console.log('Adobe callback registered successfully');
        return true;
      } catch (error) {
        console.warn('Adobe callback registration failed:', error);
        return false;
      }
    };

    const setupPolling = () => {
      let lastSelection = '';
      let selectionTimeout = null;
      let pollCount = 0;

      const checkSelection = () => {
        pollCount++;
        try {
          let selectedText = '';

          const mainSelection = window.getSelection();
          if (mainSelection && mainSelection.toString().trim()) {
            selectedText = mainSelection.toString().trim();
          }

          if (!selectedText) {
            const iframes = document.querySelectorAll('iframe');
            for (const iframe of iframes) {
              try {
                const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
                if (iframeDoc) {
                  const iframeSelection = iframeDoc.getSelection();
                  if (iframeSelection && iframeSelection.toString().trim()) {
                    selectedText = iframeSelection.toString().trim();
                    break;
                  }
                }
              } catch (e) {}
            }
          }

          if (!selectedText) {
            const adobeContainer = viewerRef.current;
            if (adobeContainer) {
              const selectors = [
                '[data-selected="true"]', '.selected-text', '.highlight',
                '.text-selection', '[aria-selected="true"]', '.adobe-selection',
                '[data-highlight]', '.pdf-selection', '.adobedc-selection'
              ];

              for (const selector of selectors) {
                const elements = adobeContainer.querySelectorAll(selector);
                if (elements.length > 0) {
                  selectedText = Array.from(elements)
                    .map(el => el.textContent || el.innerText || el.getAttribute('data-text'))
                    .filter(text => text && text.trim())
                    .join(' ')
                    .trim();

                  if (selectedText) {
                    break;
                  }
                }
              }
            }
          }

          if (selectedText &&
              selectedText !== lastSelection &&
              selectedText.length > 2 &&
              selectedText.length < 5000) {

            lastSelection = selectedText;

            if (selectionTimeout) clearTimeout(selectionTimeout);
            selectionTimeout = setTimeout(() => {
              onTextSelect(selectedText);
            }, 300);
          }
        } catch (error) {
          if (pollCount % 50 === 0) {
            console.warn(`Polling error (poll #${pollCount}):`, error);
          }
        }
      };

      let currentInterval = 100;
      pollInterval = setInterval(() => {
        checkSelection();

        if (pollCount > 50 && currentInterval < 500) {
          clearInterval(pollInterval);
          currentInterval = 500;
          pollInterval = setInterval(checkSelection, currentInterval);
        }
      }, currentInterval);

      console.log('Enhanced text selection polling started');
    };

    const setupDOMListeners = () => {
      const container = viewerRef.current;
      if (!container) return;

      let lastMouseUpTime = 0;
      let lastSelectionText = '';

      const handleMouseUp = (e) => {
        const currentTime = Date.now();
        if (currentTime - lastMouseUpTime < 100) return;
        lastMouseUpTime = currentTime;

        setTimeout(() => {
          let selectedText = '';

          const methods = [
            () => window.getSelection()?.toString()?.trim(),
            () => document.getSelection()?.toString()?.trim(),
            () => {
              const target = e.target;
              if (target && target.tagName === 'IFRAME') {
                try {
                  const iframeDoc = target.contentDocument || target.contentWindow?.document;
                  return iframeDoc?.getSelection()?.toString()?.trim();
                } catch (ex) {
                  return '';
                }
              }
              return '';
            }
          ];

          for (const method of methods) {
            try {
              selectedText = method() || '';
              if (selectedText && selectedText.length > 2) {
                break;
              }
            } catch (error) {}
          }

          if (selectedText &&
              selectedText !== lastSelectionText &&
              selectedText.length > 2 &&
              selectedText.length < 5000) {

            lastSelectionText = selectedText;
            onTextSelect(selectedText);
          }
        }, 150);
      };

      const handleSelectionChange = () => {
        setTimeout(() => {
          const selection = document.getSelection();
          const selectedText = selection ? selection.toString().trim() : '';

          if (selectedText &&
              selectedText !== lastSelectionText &&
              selectedText.length > 2 &&
              selectedText.length < 5000) {

            lastSelectionText = selectedText;
            onTextSelect(selectedText);
          }
        }, 100);
      };

      const handleKeyUp = (e) => {
        if (e.shiftKey || e.ctrlKey || e.metaKey) {
          setTimeout(() => {
            const selection = document.getSelection();
            const selectedText = selection ? selection.toString().trim() : '';

            if (selectedText &&
                selectedText !== lastSelectionText &&
                selectedText.length > 2 &&
                selectedText.length < 5000) {

              lastSelectionText = selectedText;
              onTextSelect(selectedText);
            }
          }, 100);
        }
      };

      const eventOptions = { passive: true, capture: true };

      container.addEventListener('mouseup', handleMouseUp, eventOptions);
      container.addEventListener('keyup', handleKeyUp, eventOptions);
      document.addEventListener('selectionchange', handleSelectionChange, { passive: true });
      document.addEventListener('mouseup', handleMouseUp, { passive: true });

      return () => {
        container.removeEventListener('mouseup', handleMouseUp, eventOptions);
        container.removeEventListener('keyup', handleKeyUp);
        document.removeEventListener('selectionchange', handleSelectionChange);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    };

    const adobeCallbackSuccess = registerAdobeCallback();
    const domCleanup = setupDOMListeners();
    setupPolling();

    return () => {
      if (pollInterval) clearInterval(pollInterval);
      if (domCleanup) domCleanup();
    };
  }, [onTextSelect]);

  /* --------------------------------------------------------------------------
     Error suppression for noisy Adobe SDK errors
     -------------------------------------------------------------------------- */

     const suppressAdobeErrors = useCallback(() => {
      const originalError = console.error;
      console.error = (...args) => {
        const message = args.join(' ');
        if (
          message.includes('documentViewUpdateEventData') ||
          message.includes('GET_FEATURE_FLAG') ||
          message.includes('Feature flag') ||
          message.includes('EventHandlerService') ||
          message.includes('Unexpected token') ||
          message.includes('<!doctype') ||
          message.includes('is not valid JSON') ||
          message.includes('JSON.parse')
        ) {
          return;
        }
        originalError.apply(console, args);
      };
    
      const originalWindowOnError = window.onerror;
      window.onerror = function(message, source, lineno, colno, error) {
        try {
          const msg = typeof message === 'string' ? message : (error && error.message) || '';
          if (msg && (
            msg.toString().includes('documentViewUpdateEventData') ||
            msg.toString().includes('Unexpected token') ||
            msg.toString().includes('<!doctype') ||
            msg.toString().includes('is not valid JSON')
          )) {
            return true; // suppress
          }
        } catch (e) {}
        if (typeof originalWindowOnError === 'function') {
          try { return originalWindowOnError(message, source, lineno, colno, error); } catch(e) {}
        }
        return false;
      };
    
      window.addEventListener('unhandledrejection', (ev) => {
        try {
          const reason = ev?.reason?.toString?.() || '';
          if (reason.includes('GET_FEATURE_FLAG') || 
              reason.includes('documentViewUpdateEventData') ||
              reason.includes('Unexpected token') ||
              reason.includes('<!doctype') ||
              reason.includes('is not valid JSON')) {
            ev.preventDefault();
          }
        } catch (e) {}
      });
    
      if (window.AdobeDC && window.AdobeDC.View) {
        const originalRegisterCallback = window.AdobeDC.View.prototype.registerCallback;
        window.AdobeDC.View.prototype.registerCallback = function(callbackType, callback, config) {
          try {
            return originalRegisterCallback.call(this, callbackType, callback, config);
          } catch (error) {
            const errorMsg = error && error.message || '';
            if (errorMsg.includes('Unexpected token') || 
                errorMsg.includes('<!doctype') || 
                errorMsg.includes('is not valid JSON')) {
              // Silently suppress JSON parsing errors from Adobe SDK
              return null;
            }
            console.warn('Adobe callback registration error suppressed:', errorMsg);
            return null;
          }
        };
      }
    
      if (!window.adobeFeatureFlagHandler) {
        window.adobeFeatureFlagHandler = () => false;
      }
    }, []);
  /* --------------------------------------------------------------------------
     Initialize viewer with enableAnnotationAPIs disabled (annotations removed)
     -------------------------------------------------------------------------- */

  useEffect(() => {
    let isMounted = true;
    let cleanupTextSelection = null;

    const initializeViewer = async () => {
      if (!window.AdobeDC || !viewerRef.current || !isMounted) return;

      try {
        console.log('Initializing Adobe PDF viewer...');
        suppressAdobeErrors();

        const adobeDCView = new window.AdobeDC.View({
          clientId: ADOBE_CLIENT_ID,
          divId: viewerRef.current.id,
        });

        adobeViewRef.current = adobeDCView;

        const fileData = await filePromise;
        if (!isMounted) return;

        // stable metadata id to bind metadata / future features
        const docMetadataId = docIdRef.current || `pdf_${Date.now()}_${Math.random().toString(36).slice(2,9)}`;
        docIdRef.current = docMetadataId;
        console.log('Using metadata id for preview:', docMetadataId);

        const baseConfig = {
          embedMode: 'SIZED_CONTAINER',
          focusOnLoad: false,
          showLeftHandPanel: false,
          showAnnotationTools: false,
          enableFormFilling: false,
          showDownloadPDF: false,
          showPrintPDF: false,
          showBookmarks: false,
          showThumbnails: false,
          showSearchAPITool: false,
          enablePDFAnalytics: false,
          enableAutoRotation: false,
          enableTextSelection: true,
          showAnnotationGuidelines: false,
          enableMultipleSelection: false,
          // NOTE: annotation APIs intentionally not used (highlighting removed)
          enableAnnotationAPIs: false
        };

        if (theme) {
          baseConfig.uiTheme = theme === 'light' ? 'LIGHT_GRAY' : 'DARK';
        }

        const previewOptions = {
          content: { promise: Promise.resolve(fileData) },
          metaData: { fileName: fileName || 'document.pdf', id: docMetadataId }
        };

        const adobeViewer = await adobeDCView.previewFile(previewOptions, baseConfig);
        if (!isMounted) return;

        console.log('PDF loaded successfully');
        setViewerLoaded(true);
        setIsReady(true);

        // Try to get APIs & expose enhanced wrapper (navigation-focused)
        try {
          const apis = await adobeViewer.getAPIs();
          console.log('Available Adobe APIs:', Object.keys(apis || {}));

          const enhancedAPIs = {
            ...apis,
            navigateToPage: async (pageNumber) => {
              try {
                if (apis && typeof apis.gotoLocation === 'function') {
                  try {
                    const result = await apis.gotoLocation(pageNumber);
                    if (result !== false && result !== null) return result;
                  } catch (e) { /* ignore */ }
                }
                if (apis && typeof apis.goToPage === 'function') {
                  try {
                    const result = await apis.goToPage(pageNumber);
                    if (result !== false && result !== null) return result;
                  } catch (e) { /* ignore */ }
                }
                if (adobeViewer && typeof adobeViewer.gotoLocation === 'function') {
                  try {
                    const result = await adobeViewer.gotoLocation(pageNumber);
                    if (result !== false && result !== null) return result;
                  } catch (e) { /* ignore */ }
                }
                return false;
              } catch (error) {
                console.error('Navigation failed with error:', error);
                return false;
              }
            },
            getCurrentPage: async () => {
              try {
                if (apis && typeof apis.getCurrentPage === 'function') {
                  return await apis.getCurrentPage();
                }
                return 1;
              } catch (error) {
                console.error('Get current page failed:', error);
                return 1;
              }
            },
            getTotalPages: async () => {
              try {
                if (apis && typeof apis.getPageCount === 'function') {
                  return await apis.getPageCount();
                } else if (apis && typeof apis.getNumPages === 'function') {
                  return await apis.getNumPages();
                }
                return 1;
              } catch (error) {
                console.error('Get page count failed:', error);
                return 1;
              }
            },
            getRawViewer: () => adobeViewer,
            getRawAPIs: () => apis,
            checkAPIAvailability: () => {
              const availableAPIs = apis ? Object.keys(apis) : [];
              return {
                hasNavigation: availableAPIs.includes('gotoLocation'),
                hasSearch: availableAPIs.includes('search'),
                hasBookmarks: availableAPIs.includes('getBookmarks') || availableAPIs.includes('getOutline'),
                allMethods: availableAPIs
              };
            }
          };

          enhancedAPIsRef.current = enhancedAPIs;

          if (isMounted && onViewerReady) {
            onViewerReady(enhancedAPIs);
          }

          if (pageNumber && pageNumber > 1) {
            setTimeout(() => {
              enhancedAPIs.navigateToPage(pageNumber)
                .then(result => console.log(`Initial navigation to page ${pageNumber} result:`, result))
                .catch(error => console.warn(`Initial navigation to page ${pageNumber} failed:`, error));
            }, 1200);
          }
        } catch (apiError) {
          console.warn('Could not get Adobe APIs:', apiError);
          if (isMounted && onViewerReady) {
            onViewerReady({
              navigateToPage: async (pageNumber) => { console.warn('APIs unavailable navigateToPage'); return false; },
              getRawViewer: () => adobeViewer,
              checkAPIAvailability: () => ({ error: 'APIs unavailable' })
            });
          }
        }

        // register some debug callbacks (navigation events)
        try {
          adobeDCView.registerCallback(
            window.AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
            (event) => {
              if (['PAGE_VIEW', 'BOOKMARK_ITEM_CLICK', 'THUMBNAIL_CLICK'].includes(event.type)) {
                console.log('Adobe Navigation Event:', event.type, event.data);
              }
            },
            { enableFilePreviewEvents: true, enablePDFAnalytics: false }
          );
        } catch (callbackError) {
          console.warn('Could not register navigation callbacks:', callbackError);
        }

        // delay then setup text selection wiring
        setTimeout(async () => {
          if (isMounted) {
            try {
              cleanupTextSelection = await setupTextSelection(adobeDCView, adobeViewer);
            } catch (textSelectionError) {
              console.warn('Text selection setup failed:', textSelectionError);
            }
          }
        }, 1400);

        adobeViewerRef.current = adobeViewer;
      } catch (error) {
        if (isMounted) {
          console.error("Error rendering PDF with Adobe SDK:", error);
        }
      }
    };

    if (window.AdobeDC) {
      initializeViewer();
    } else {
      const handleSDKReady = () => {
        if (isMounted) initializeViewer();
      };
      document.addEventListener('adobe_dc_view_sdk.ready', handleSDKReady);
      return () => {
        isMounted = false;
        document.removeEventListener('adobe_dc_view_sdk.ready', handleSDKReady);
        if (cleanupTextSelection) cleanupTextSelection();
      };
    }

    return () => {
      isMounted = false;
      setIsReady(false);
      setViewerLoaded(false);

      if (cleanupTextSelection) cleanupTextSelection();
      if (adobeViewRef.current) adobeViewRef.current = null;
      if (adobeViewerRef.current) adobeViewerRef.current = null;
      enhancedAPIsRef.current = null;
    };
  }, [filePromise, fileName, onViewerReady, setupTextSelection, suppressAdobeErrors, theme, pageNumber]);

  /* --------------------------------------------------------------------------
     Navigate on prop pageNumber changes
     -------------------------------------------------------------------------- */

  useEffect(() => {
    if (pageNumber && enhancedAPIsRef.current && enhancedAPIsRef.current.navigateToPage) {
      console.log(`📍 Navigating to page ${pageNumber} via prop change`);
      enhancedAPIsRef.current.navigateToPage(pageNumber)
        .then(result => {
          console.log(`Navigation to page ${pageNumber} result:`, result);
        })
        .catch(error => {
          console.warn(`Navigation to page ${pageNumber} failed:`, error);
        });
    }
  }, [pageNumber]);

  const viewerId = `adobe-dc-view-${React.useId()}`;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div
        id={viewerId}
        ref={viewerRef}
        style={{
          width: '100%',
          height: '100%',
          opacity: isReady ? 1 : 0.5,
          transition: 'opacity 0.3s ease'
        }}
      />
      {!isReady && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          color: theme === 'light' ? '#666' : '#ccc'
        }}>
          Loading PDF viewer...
        </div>
      )}
      {isReady && !viewerLoaded && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          padding: '8px 12px',
          background: theme === 'light' ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
          borderRadius: '4px',
          fontSize: '12px',
          color: theme === 'light' ? '#666' : '#ccc'
        }}>
          Initializing navigation and thumbnails...
        </div>
      )}
    </div>
  );
});

export default PDFViewer;
