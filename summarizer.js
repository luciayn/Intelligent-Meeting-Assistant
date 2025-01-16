export async function summarizeTextWithHF(text) {
    const hfToken = ""; // CADA UNO PONE SU TOKEN     // en producción, ¡usa variable de entorno en tu servidor!
    const url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn";
  
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${hfToken}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ inputs: text })
      });
  
      const data = await response.json();
      if (data && data[0]?.summary_text) {
        return data[0].summary_text.trim();
      } else {
        console.error("No se pudo obtener resumen de la respuesta", data);
        return "";
      }
    } catch (error) {
      console.error("Error al comunicarse con Hugging Face:", error);
      return "";
    }
  }
  