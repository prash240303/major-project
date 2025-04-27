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

  const styles = {
    container: {
      fontFamily: "'Inter', sans-serif",
      maxWidth: "1200px",
      margin: "0 auto",
      padding: "0 24px",
    },
    header: {
      fontSize: "28px",
      fontWeight: "700",
      color: "#1E40AF",
      padding: "24px 0",
      borderBottom: "1px solid #E5E7EB",
      marginBottom: "24px",
      display: "flex",
      alignItems: "center",
      gap: "12px",
      fontFamily: "'Inter', sans-serif",
    },
    logo: {
      fontSize: "32px",
      marginRight: "8px",
    },
    mainContent: {
      padding: "24px 0",
      fontFamily: "'Inter', sans-serif",
    },
    welcome: {
      fontSize: "18px",
      color: "#4B5563",
      marginBottom: "36px",
      lineHeight: "1.6",
      fontFamily: "'Inter', sans-serif",
    },
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>
        <span style={styles.logo}>
          <img
            src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
            alt="Logo"
            style={{
              width: "40px",
              height: "40px",
              borderRadius: "50%",
              marginRight: "8px",
            }}
          />
        </span>
        Welcome to Markdarshak
      </h1>

      <div style={styles.mainContent}>
        <p style={styles.welcome}>
          Your intelligent AI assistant for all your questions and needs. Click
          on the chat bubble in the bottom right corner to get started.
        </p>
      </div>

      <Chatbot />
    </div>
  );
}

export default App;
