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
});