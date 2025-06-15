import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Chatbot from './Chatbot'

// Component to handle font loading for embedded chatbot
function EmbeddedChatbot() {
  // Add Inter font to the document if it's not already added
  useEffect(() => {
    const link = document.createElement("link");
    link.href =
      "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    return () => {
      // Only remove if it still exists (prevents errors in embedded contexts)
      if (document.head.contains(link)) {
        document.head.removeChild(link);
      }
    };
  }, []);

  return (
    <div className="font-inter">
      <Chatbot />
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <EmbeddedChatbot />
  </StrictMode>,
)