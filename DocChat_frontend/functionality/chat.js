const chatBox = document.getElementById("chat-window");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send");

// Basic responses
const responses = {
  "hi": "Hello! How can I assist you today?",
  "hii": "Hello! How can I assist you today?",
  "hello": "Hi there! What can I do for you?",
  "hlo": "Hi there! What can I do for you?",
  "how are you": "I'm just a bot, but I'm doing great!",
  "thank you": "your most welcome!",
  "tq": "your most welcome!",
  "bye": "Goodbye! Have a nice day!",
};

// Function to add message to chat
function addMessage(message, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.textContent = message;
  msgDiv.className = sender === "bot" ? "bot-message" : "user-message";
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

// Function to get bot response
function getBotResponse(input) {
  input = input.toLowerCase();
  for (let key in responses) {
    if (input.includes(key)) {
      return responses[key];
    }
  }
  return "Sorry, I didn't understand that.";
}

// Event listeners
sendBtn.addEventListener("click", () => {
  const message = userInput.value.trim();
  if (message !== "") {
    addMessage(message, "user");
    const botReply = getBotResponse(message);
    setTimeout(() => addMessage(botReply, "bot"), 500); // Delay for realism
    userInput.value = "";
  }
});

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendBtn.click();
});
