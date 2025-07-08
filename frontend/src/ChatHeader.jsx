import { motion } from "framer-motion";

const ChatHeader = ({ toggleChat }) => {
  return (
    <motion.div
      className="flex-shrink-0 px-6 py-4 bg-blue-50 border-b border-blue-200"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <div className="max-w-4xl mx-auto flex justify-between items-center">
        <div className="flex items-center gap-3">
          <motion.img
            src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
            alt="Logo"
            className="w-8 h-8 rounded-full"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 500, damping: 15 }}
          />
          <span className="text-blue-900 font-semibold text-lg">
            Margdarshak Assistant
          </span>
        </div>
        <motion.button
          onClick={toggleChat}
          className="w-10 h-10 flex items-center justify-center rounded-full text-gray-500 hover:bg-blue-100 hover:text-gray-600 transition text-xl"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          ✕
        </motion.button>
      </div>
    </motion.div>
  );
};

export default ChatHeader;
