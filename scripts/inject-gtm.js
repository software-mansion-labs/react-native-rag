const fs = require('fs');
const path = require('path');

const GTM_CONTAINER_ID = 'GTM-N9JBFWFP';

const docsDir = path.join(__dirname, '..', 'docs');

if (!fs.existsSync(docsDir)) {
  console.error('Error: docs directory not found. Run typedoc first.');
  process.exit(1);
}

const googleSiteVerification = `<meta name="google-site-verification" content="o1aP6mAxMf2S_-KBtSaitCHmRMHjIzIOe59MHzgTTbU" />`;

const gtmHeadScript = `<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','${GTM_CONTAINER_ID}');</script>
<!-- End Google Tag Manager -->`;

const gtmBodyScript = `<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=${GTM_CONTAINER_ID}"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->`;

function injectGTM(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  if (content.includes('googletagmanager.com/gtm.js')) {
    return;
  }

  if (content.includes('<head>')) {
    content = content.replace(
      /<head>/,
      `<head>\n${gtmHeadScript}\n${googleSiteVerification}`
    );
  }
  if (content.includes('<body>')) {
    content = content.replace(/<body>/, `<body>\n${gtmBodyScript}`);
  }

  fs.writeFileSync(filePath, content, 'utf8');
}

function processDirectory(dir) {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      processDirectory(filePath);
    } else if (file.endsWith('.html')) {
      injectGTM(filePath);
    }
  }
}

processDirectory(docsDir);
console.log('GTM container injected successfully.');
