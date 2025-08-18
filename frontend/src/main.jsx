import React from 'react';
import ReactDOM from 'react-dom/client';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import './index.css';
import { ThemeProvider } from './context/ThemeContext';

// Import your layout and views
import LandingPage from './pages/LandingPage';
import LibraryView from './pages/LibraryView';
import DocumentReaderView from './pages/DocumentReaderView';
import CollectionAnalysisPage from './pages/CollectionAnalysisPage';

const router = createBrowserRouter([
  {
    path: "/",
    children: [
      {
        index: true,
        element: <LandingPage />,
      },
      {
        path: "library",
        element: <LibraryView />,
      },
      {
        path: "reader",
        element: <DocumentReaderView />,
      },
      {
        path: "collection-analysis",
        element: <CollectionAnalysisPage />,
      },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  </React.StrictMode>
);