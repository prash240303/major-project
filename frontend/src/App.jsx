import React, { useEffect } from "react";
import Chatbot from "./Chatbot";

function App() {
  // Add Inter font to the document if it's not already added
  useEffect(() => {
    const link = document.createElement("link");
    link.href =
      "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    return () => {
      document.head.removeChild(link);
    };
  }, []);

  return (
    <div className="font-inter max-w-[1200px] mx-auto px-6">
      <h1 className="text-2xl font-bold text-blue-800 py-6 border-b border-gray-200 mb-6 flex items-center gap-3">
        <span>
          <img
            src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
            alt="Logo"
            className="w-10 h-10 rounded-full mr-2"
          />
        </span>
        Welcome to Margdarshak
      </h1>

      <div className="py-6">
        <div className="text-lg text-gray-600 mb-9 leading-relaxed">
          Hello! I'm your NITJ Margdarshak assistant. How can I help you today?
        </div>
      </div>

      <Chatbot />
    </div>
  );
}

export default App;
