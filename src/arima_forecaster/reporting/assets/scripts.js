document.addEventListener('DOMContentLoaded', function() {
  // Image zoom functionality
  const images = document.querySelectorAll('img');
  
  // Create modal container
  const modal = document.createElement('div');
  modal.className = 'image-modal';
  modal.innerHTML = '<span class="close-modal">&times;</span><img class="modal-content"><div class="image-caption"></div>';
  document.body.appendChild(modal);
  
  const modalImg = modal.querySelector('.modal-content');
  const captionText = modal.querySelector('.image-caption');
  const closeBtn = modal.querySelector('.close-modal');
  
  images.forEach(img => {
    // Skip small images (icons, etc.)
    if (img.width > 100) {
      // Wrap image in container
      const container = document.createElement('div');
      container.className = 'image-container';
      container.style.position = 'relative';
      img.parentNode.insertBefore(container, img);
      container.appendChild(img);
      
      // Add zoomable class
      img.classList.add('zoomable-image');
      
      // Create magnifying lens
      const lens = document.createElement('div');
      lens.className = 'magnifier-lens';
      container.appendChild(lens);
      
      // Create magnified result container (usando background-image)
      const result = document.createElement('div');
      result.className = 'magnifier-result';
      // Imposta l'immagine come background invece di elemento img interno
      result.style.backgroundImage = `url('${img.src}')`;
      container.appendChild(result);
      
      // Create download icon
      const downloadIcon = document.createElement('div');
      downloadIcon.className = 'download-icon';
      downloadIcon.innerHTML = '💾';
      downloadIcon.title = 'Scarica immagine';
      container.appendChild(downloadIcon);
      
      // Calculate zoom level
      const zoom = 3;
      
      // Variabili per gestire il ritardo e il controllo della lente
      let lensTimeout = null;
      let isLensVisible = false;
      
      // Set up magnifier functionality con miglioramenti
      function magnify(e) {
        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Debug coordinate e dimensioni
        console.log('Mouse coordinates:', { x, y, imgWidth: img.width, imgHeight: img.height });
        console.log('Image loading state:', { complete: img.complete, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight });
        
        // Check if mouse is within image bounds
        if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
          hideLens();
          return;
        }
        
        // Definisci margini per nascondere la lente vicino ai bordi
        const edgeMargin = 60; // pixels dal bordo
        const lensRadius = 40; // metà della dimensione della lente
        
        // Nascondi la lente se troppo vicina ai bordi
        if (x < edgeMargin || y < edgeMargin || 
            x > rect.width - edgeMargin || y > rect.height - edgeMargin) {
          hideLens();
          return;
        }
        
        // Position lens al centro del cursore
        let lensX = x - lensRadius;
        let lensY = y - lensRadius;
        
        lens.style.left = lensX + 'px';
        lens.style.top = lensY + 'px';
        
        // Position result sempre fuori dall'immagine
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const imgRight = rect.right;
        const imgLeft = rect.left;
        
        let resultX, resultY;
        
        // Prova a posizionare a destra dell'immagine
        if (imgRight + result.offsetWidth + 20 < viewportWidth) {
          resultX = imgRight + 20;
          resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
        }
        // Altrimenti a sinistra dell'immagine
        else if (imgLeft - result.offsetWidth - 20 > 0) {
          resultX = imgLeft - result.offsetWidth - 20;
          resultY = rect.top + (rect.height / 2) - (result.offsetHeight / 2);
        }
        // Come fallback, posiziona sopra l'immagine se possibile
        else if (rect.top - result.offsetHeight - 20 > 0) {
          resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
          resultY = rect.top - result.offsetHeight - 20;
        }
        // Ultimo fallback: sotto l'immagine
        else {
          resultX = rect.left + (rect.width / 2) - (result.offsetWidth / 2);
          resultY = rect.bottom + 20;
        }
        
        // Assicura che il risultato rimanga all'interno del viewport
        resultX = Math.max(10, Math.min(resultX, viewportWidth - result.offsetWidth - 10));
        resultY = Math.max(10, Math.min(resultY, viewportHeight - result.offsetHeight - 10));
        
        result.style.left = resultX + 'px';
        result.style.top = resultY + 'px';
        
        // Assicurati che l'immagine sia completamente caricata prima di calcolare
        if (!img.complete || img.naturalWidth === 0) {
          console.log('Immagine non completamente caricata, aspettando...');
          img.onload = function() {
            magnify(e); // Richiama la funzione quando l'immagine è caricata
          };
          return;
        }
        
        // Calculate magnified image position con correzioni
        const scaleX = img.naturalWidth / rect.width;
        const scaleY = img.naturalHeight / rect.height;
        
        const bgX = x * scaleX;
        const bgY = y * scaleY;
        
        console.log('Natural dimensions:', { 
          naturalWidth: img.naturalWidth, 
          naturalHeight: img.naturalHeight,
          scaleX, scaleY, bgX, bgY 
        });
        
        // Calcola posizionamento background con approccio più affidabile
        const magnifiedWidth = img.naturalWidth * zoom;
        const magnifiedHeight = img.naturalHeight * zoom;
        
        // Posizione del background per centrare l'area ingrandita
        const bgPosX = -(bgX * zoom - result.offsetWidth / 2);
        const bgPosY = -(bgY * zoom - result.offsetHeight / 2);
        
        // Applica background properties
        result.style.backgroundImage = `url('${img.src}')`;
        result.style.backgroundSize = `${magnifiedWidth}px ${magnifiedHeight}px`;
        result.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
        result.style.backgroundRepeat = 'no-repeat';
        
        console.log('Background positioning:', {
          backgroundSize: result.style.backgroundSize,
          backgroundPosition: result.style.backgroundPosition,
          magnifiedDimensions: { width: magnifiedWidth, height: magnifiedHeight },
          centerOffset: { x: bgPosX, y: bgPosY },
          mousePosition: { bgX, bgY }
        });
        
        // Mostra lente e risultato con ritardo se non già visibili
        if (!isLensVisible) {
          showLensWithDelay();
        }
      }
      
      function showLensWithDelay() {
        clearTimeout(lensTimeout);
        lensTimeout = setTimeout(() => {
          lens.style.display = 'block';
          result.style.display = 'block';
          // Forza reflow prima di applicare opacity
          lens.offsetHeight;
          result.offsetHeight;
          // Background-image è già impostato, non servono controlli aggiuntivi
          lens.style.opacity = '1';
          result.style.opacity = '1';
          isLensVisible = true;
          console.log('Lens and result shown, result dimensions:', { width: result.offsetWidth, height: result.offsetHeight });
        }, 200); // Ritardo di 200ms
      }
      
      function hideLens() {
        clearTimeout(lensTimeout);
        lens.style.opacity = '0';
        result.style.opacity = '0';
        setTimeout(() => {
          lens.style.display = 'none';
          result.style.display = 'none';
          isLensVisible = false;
        }, 200); // Tempo per l'animazione di fade out
      }
      
      // Add mouse events
      img.addEventListener('mousemove', magnify);
      img.addEventListener('mouseenter', magnify);
      img.addEventListener('mouseleave', function() {
        hideLens();
      });
      
      // Add download functionality
      downloadIcon.addEventListener('click', function(e) {
        e.stopPropagation(); // Prevent triggering modal
        downloadImage(img);
      });
      
      // Add click event for modal
      img.addEventListener('click', function() {
        modal.style.display = 'block';
        modalImg.src = this.src;
        
        // Preserve aspect ratio by setting natural dimensions
        modalImg.onload = function() {
          const aspectRatio = this.naturalWidth / this.naturalHeight;
          const maxWidth = window.innerWidth * 0.9;
          const maxHeight = window.innerHeight * 0.9;
          
          let width, height;
          
          if (aspectRatio > maxWidth / maxHeight) {
            // Image is wider - limit by width
            width = maxWidth;
            height = maxWidth / aspectRatio;
          } else {
            // Image is taller - limit by height
            height = maxHeight;
            width = maxHeight * aspectRatio;
          }
          
          this.style.width = width + 'px';
          this.style.height = height + 'px';
          this.style.maxWidth = 'none';
          this.style.maxHeight = 'none';
        };
        
        captionText.innerHTML = this.alt || 'Clicca fuori dall\'immagine o premi ESC per chiudere';
      });
    }
  });
  
  // Close modal on click
  closeBtn.addEventListener('click', function() {
    modal.style.display = 'none';
  });
  
  modal.addEventListener('click', function(e) {
    if (e.target === modal) {
      modal.style.display = 'none';
    }
  });
  
  // Close modal on ESC key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && modal.style.display === 'block') {
      modal.style.display = 'none';
    }
  });
  
  // Add active highlighting to TOC based on scroll position
  const sections = document.querySelectorAll('section[id]');
  const tocLinks = document.querySelectorAll('#TOC a[href^="#"]');
  
  let lastActiveSection = null; // Track the last activated section to prevent redundant updates
  
  function highlightTOC() {
    let current = '';
    const scrollY = window.pageYOffset;
    const offset = 100;
    const viewportHeight = window.innerHeight;
    
    // Conservative approach: only change if a section is clearly dominant
    let bestSection = null;
    let maxVisibility = 0;
    
    sections.forEach(section => {
      const rect = section.getBoundingClientRect();
      const sectionTop = rect.top;
      const sectionBottom = rect.bottom;
      
      // Calculate what percentage of the viewport this section occupies
      const visibleTop = Math.max(0, sectionTop);
      const visibleBottom = Math.min(viewportHeight, sectionBottom);
      const visibleHeight = Math.max(0, visibleBottom - visibleTop);
      const visibilityRatio = visibleHeight / viewportHeight;
      
      // Only consider sections that take up significant space (at least 25% of viewport)
      // AND prefer sections that are positioned near the top
      if (visibilityRatio >= 0.25 && sectionTop <= viewportHeight * 0.3) {
        // Add hysteresis: require a bigger difference to switch from current section
        const isCurrentSection = section.getAttribute('id') === lastActiveSection;
        const hysteresis = isCurrentSection ? -0.1 : 0; // 10% bonus to current section
        
        if (visibilityRatio + hysteresis > maxVisibility) {
          maxVisibility = visibilityRatio;
          bestSection = section;
        }
      }
    });
    
    if (bestSection) {
      current = bestSection.getAttribute('id');
    }
    
    // Fallback: if no section found, use first section at top of page
    if (!current && scrollY < 200) {
      current = sections.length > 0 ? sections[0].getAttribute('id') : '';
    }
    
    // Only update if the section actually changed
    if (current === lastActiveSection) {
      return; // No change, skip update
    }
    
    lastActiveSection = current;
    
    // Update TOC highlighting - remove active from all links first
    tocLinks.forEach(link => {
      const parent = link.parentElement;
      parent.classList.remove('active');
      
      // Also remove override classes from children
      const childLinks = parent.querySelectorAll('ul li');
      childLinks.forEach(childLi => {
        childLi.classList.remove('toc-child-override');
      });
    });
    
    // Add active to current section with child override
    if (current) {
      const currentLink = document.querySelector(`#TOC a[href="#${current}"]`);
      if (currentLink) {
        const parentLi = currentLink.parentElement;
        
        // Activate the current section
        parentLi.classList.add('active');
        
        // Add override class to prevent cascading to children
        const childLinks = parentLi.querySelectorAll('ul li');
        childLinks.forEach(childLi => {
          childLi.classList.add('toc-child-override');
        });
      }
    }
  }
  
  // Throttle scroll events for better performance
  let scrollTimeout = null;
  function throttledHighlight() {
    if (scrollTimeout === null) {
      scrollTimeout = setTimeout(() => {
        highlightTOC();
        scrollTimeout = null;
      }, 500);
    }
  }
  
  window.addEventListener('scroll', throttledHighlight);
  
  // Initial call with slight delay to ensure DOM is ready
  setTimeout(highlightTOC, 100);
  
  // Function to download image
  function downloadImage(img) {
    try {
      // Create canvas to convert image
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Set canvas dimensions to match image natural size
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      
      // Draw image on canvas
      ctx.drawImage(img, 0, 0);
      
      // Convert to blob and download
      canvas.toBlob(function(blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        
        // Generate intelligent filename
        let filename = '';
        
        // Try to use alt text first (cleaned up)
        if (img.alt && img.alt.trim()) {
          filename = img.alt.trim()
            .replace(/[^a-zA-Z0-9\s\-_]/g, '') // Remove special chars except spaces, hyphens, underscores
            .replace(/\s+/g, '_') // Replace spaces with underscores
            .toLowerCase()
            .substring(0, 50); // Limit length
        }
        
        // If alt text is empty or too short, try to extract from src
        if (!filename || filename.length < 3) {
          const srcParts = img.src.split('/');
          const lastPart = srcParts[srcParts.length - 1];
          
          // Only use if it looks like a real filename (not base64 or very long)
          if (lastPart && lastPart.length < 100 && !lastPart.startsWith('data:')) {
            filename = lastPart.split('.')[0] // Remove extension
              .replace(/[^a-zA-Z0-9\s\-_]/g, '')
              .replace(/\s+/g, '_')
              .toLowerCase();
          }
        }
        
        // Fallback to descriptive names based on common chart types
        if (!filename || filename.length < 3) {
          // Try to detect chart type from context
          const container = img.closest('section, div, figure');
          if (container) {
            const textContent = container.textContent.toLowerCase();
            if (textContent.includes('forecast')) filename = 'forecast_chart';
            else if (textContent.includes('residui')) filename = 'residuals_analysis';
            else if (textContent.includes('performance')) filename = 'performance_chart';
            else if (textContent.includes('metrica')) filename = 'metrics_chart';
            else if (textContent.includes('diagnostic')) filename = 'diagnostics_plot';
            else filename = 'arima_chart';
          } else {
            filename = 'arima_chart';
          }
        }
        
        // Add timestamp to make unique
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
        filename = `${filename}_${timestamp}`;
        
        // Ensure .png extension
        if (!filename.endsWith('.png')) {
          filename += '.png';
        }
        
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        // Clean up
        URL.revokeObjectURL(url);
        
        console.log(`Image downloaded: ${filename}`);
      }, 'image/png', 1.0);
      
    } catch (error) {
      console.error('Error downloading image:', error);
      
      // Fallback: try direct download link
      const a = document.createElement('a');
      a.href = img.src;
      a.download = img.alt || 'chart.png';
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }
});