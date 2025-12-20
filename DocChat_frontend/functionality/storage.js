const pt = document.getElementById("pt");

if (pt) {
  const pdfDetails = sessionStorage.getItem("pdfDetails");
  const userText = sessionStorage.getItem("userText");
  console.log(sessionStorage.getItem("userText"))

  if (pdfDetails && userText) {
    pt.textContent = "📄 PDF & Text Uploaded...";
  }
  else if (pdfDetails) {
    pt.textContent = "📄 PDF Uploaded...";
  }
  else if (userText) {
    pt.textContent = "📝 Text Uploaded...";
  }
  else {
    pt.textContent = "⚠️ No file or text uploaded!";// Get the buttons
  }
}

//  Enable buttons logic

const showTextBtn = document.querySelector(".file-box:nth-of-type(1) button");
const showPdfBtn = document.querySelector(".file-box:nth-of-type(2) button");

if (showTextBtn && showPdfBtn) {
  const userText = sessionStorage.getItem("userText");
  const pdfDetails = sessionStorage.getItem("pdfDetails");

  // enable or disable buttons based on uploaded content
  if (userText) {
    showTextBtn.disabled = false;
  } else {
    showTextBtn.disabled = true;
  }

  if (pdfDetails) {
    showPdfBtn.disabled = false;
  } else {
    showPdfBtn.disabled = true;
  }
  const st = document.getElementById("st");
  const pt = document.getElementById("pt");

  showTextBtn.addEventListener("click", () => {
    if (st.classList.contains("hidden")) {
      st.classList.remove("hidden");
      st.innerHTML = `
      <p><strong>File Name:</strong> ${userText}</p>
      `;
      showTextBtn.textContent = "Hide";
    } else {
      st.classList.add("hidden");
      showTextBtn.textContent = "👁️Show Text";
    }
  });

  showPdfBtn.addEventListener("click", () => {
    if (pt1.classList.contains("hidden")) {
      pt1.classList.remove("hidden");
      const pdfInfo = JSON.parse(pdfDetails);
      pt1.innerHTML = `
      <p><strong>File Name:</strong> ${pdfInfo.name}</p>
      `;
      showPdfBtn.textContent = "Hide";
    } else {
      pt1.classList.add("hidden");
      showPdfBtn.textContent = "👁️Show PDF";
    }
  });

}

