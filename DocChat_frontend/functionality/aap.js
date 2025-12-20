//landing page
document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('getstarted');
  btn.addEventListener('click', function (e) {
    e.preventDefault();
    window.location.href = 'upload.html';
  });
});
document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('try');
  btn.addEventListener('click', function (e) {
    e.preventDefault();
    window.location.href = 'upload.html';
  });
});
//train page start chatting button
document.addEventListener('DOMContentLoaded', function () {
  const cbtn = document.getElementById('schat');
  cbtn.addEventListener('click', function (e) {
    e.preventDefault();
    window.location.href = 'chat.html';
  });
});

// Clear only when the current page is the upload page AND it was refreshed
if (
  window.location.pathname.includes("upload.html") &&
  performance.getEntriesByType("navigation")[0].type === "reload"
) {
  sessionStorage.removeItem("pdfDetails");
  sessionStorage.removeItem("pdfBase64");
  sessionStorage.removeItem("userText");
}

// References
const pdfInput = document.getElementById("pdfInput");
const fileDetails = document.getElementById("fileDetails");
const textInput = document.getElementById("textInput");

// Restore previously stored PDF info (if any)
window.addEventListener("DOMContentLoaded", () => {
  const storedPDF = sessionStorage.getItem("pdfDetails");
  if (storedPDF) {
    const pdfInfo = JSON.parse(storedPDF);
    fileDetails.innerHTML = `
      <p><strong>File Name:</strong> ${pdfInfo.name}</p>
      <p><strong>File Type:</strong> ${pdfInfo.type}</p>
      <p><strong>File Size:</strong> ${pdfInfo.size} KB</p>
    `;
  }
});

// Handle PDF upload and store
pdfInput.addEventListener("change", function () {
  if (pdfInput.files.length > 0) {
    const file = pdfInput.files[0];
    const sizeKB = (file.size / 1024).toFixed(2); // bytes → KB

    fileDetails.innerHTML = `
      <p><strong>File Name:</strong> ${file.name}</p>
      <p><strong>File Type:</strong> ${file.type}</p>
      <p><strong>File Size:</strong> ${sizeKB} KB</p>
    `;

    const pdfInfo = {
      name: file.name,
      type: file.type,
      size: sizeKB,
    };
    sessionStorage.setItem("pdfDetails", JSON.stringify(pdfInfo));

    const reader = new FileReader();
    reader.onload = function (e) {
      sessionStorage.setItem("pdfBase64", e.target.result);
    };
    reader.readAsDataURL(file);
  } else {
    fileDetails.innerHTML = "";
    sessionStorage.removeItem("pdfDetails");
    sessionStorage.removeItem("pdfBase64");
  }
});

// Clear text input manually
const clearBtn = document.getElementById("Clean");
clearBtn.addEventListener("click", function () {
  textInput.value = "";
  sessionStorage.removeItem("userText");
});
//save data and go to next page
function saveAndRedirect() {
  const text = textInput.value.trim();
  const len = text.length;

  if (text === "" && pdfInput.files.length === 0) {
    alert("Please enter text first! or upload a PDF first");
  } else if (len > 50000) {
    alert("🚫 Beyond the word limit (max 50,000 characters)!");
  } else {
    sessionStorage.setItem("userText", text);
    window.location.href = "train.html";
  }
}

document.getElementById("train1").addEventListener("click", saveAndRedirect);
document.getElementById("train2").addEventListener("click", saveAndRedirect);
