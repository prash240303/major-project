import React from "react";
import { motion } from "framer-motion";

const ChatInput = ({ prompt, setPrompt, sendMessage, handleKeyDown, loading }) => {
  return (
    <motion.div
      className="flex-shrink-0 px-6 py-6 border-t border-gray-200 bg-white"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="max-w-4xl mx-auto">
        <motion.div
          className="text-xs text-gray-600 mb-4 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.8 }}
          transition={{ delay: 0.4 }}
        >
          <span className="font-semibold text-sm text-red-500">Disclaimer:</span>{" "}
          This assistant provides general guidance and is not a substitute for official academic or administrative advice.
        </motion.div>

        <div className="flex gap-4">
          <motion.input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything..."
            disabled={loading}
            className={`flex-1 px-4 py-3 border rounded-full text-sm outline-none transition ${
              loading ? "bg-gray-100 text-gray-500" : "bg-white"
            }`}
            whileFocus={{ boxShadow: "0 0 0 3px rgba(59, 130, 246, 0.3)" }}
          />

          <motion.button
            onClick={sendMessage}
            disabled={loading || !prompt.trim()}
            className={`px-6 py-3 rounded-full text-base shadow transition ${
              loading || !prompt.trim()
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-blue-500 text-white hover:bg-blue-600"
            }`}
            whileHover={
              !loading && prompt.trim()
                ? { scale: 1.05, y: -2 }
                : {}
            }
            whileTap={!loading && prompt.trim() ? { scale: 0.95 } : {}}
          >
            {loading ? "..." : "Send"}
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default ChatInput;
