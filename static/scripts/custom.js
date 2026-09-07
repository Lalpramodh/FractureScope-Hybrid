document.addEventListener('DOMContentLoaded', function () {
  var year = document.querySelector('#displayYear');
  if (year) year.textContent = new Date().getFullYear();

  var toggle = document.querySelector('[data-menu-toggle]');
  var menu = document.querySelector('[data-menu]');
  if (toggle && menu) {
    toggle.addEventListener('click', function () {
      var open = menu.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  document.querySelectorAll('[data-dismiss]').forEach(function (button) {
    button.addEventListener('click', function () {
      var flash = button.closest('.flash');
      if (flash) flash.remove();
    });
  });

  var fileInput = document.querySelector('[data-file-input]');
  var fileName = document.querySelector('[data-file-name]');
  if (fileInput && fileName) {
    fileInput.addEventListener('change', function () {
      fileName.textContent = fileInput.files.length ? fileInput.files[0].name : 'PNG, JPG or JPEG up to 16 MB';
    });
  }

  var chatToggle = document.querySelector('[data-chat-toggle]');
  var chatClose = document.querySelector('[data-chat-close]');
  var chatPanel = document.querySelector('[data-chat-panel]');
  var chatForm = document.querySelector('[data-chat-form]');
  var chatInput = document.querySelector('[data-chat-input]');
  var chatMessages = document.querySelector('[data-chat-messages]');
  var chatHistory = [];
  function setChatOpen(open) {
    if (!chatPanel || !chatToggle) return;
    chatPanel.hidden = !open;
    chatToggle.setAttribute('aria-expanded', String(open));
    if (open && chatInput) chatInput.focus();
  }
  function addChatMessage(text, role) {
    var message = document.createElement('div');
    message.className = 'chat-message chat-message-' + role;
    message.textContent = text;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  if (chatToggle && chatPanel && chatForm) {
    chatToggle.addEventListener('click', function () { setChatOpen(chatPanel.hidden); });
    if (chatClose) chatClose.addEventListener('click', function () { setChatOpen(false); });
    chatForm.addEventListener('submit', async function (event) {
      event.preventDefault();
      var text = chatInput.value.trim();
      if (!text) return;
      chatInput.value = '';
      chatHistory.push({ role: 'user', content: text });
      addChatMessage(text, 'user');
      var waiting = 'Thinking...';
      addChatMessage(waiting, 'assistant');
      var pending = chatMessages.lastElementChild;
      try {
        var response = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: chatHistory }) });
        var data = await response.json();
        pending.textContent = data.reply || 'I could not answer that right now.';
        chatHistory.push({ role: 'assistant', content: pending.textContent });
      } catch (error) {
        pending.textContent = 'Chat is temporarily unavailable. Please try again later.';
      }
    });
  }
});