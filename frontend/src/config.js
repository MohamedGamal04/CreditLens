/* Backend API base URL, chosen by where the frontend is served from:
   - localhost / 127.0.0.1  -> "" (same origin: local `make serve`)
   - *.hf.space             -> "" (same origin, if ever served by the HF backend)
   - anything else (Vercel) -> the deployed HF Spaces backend
*/
window.API_BASE = (
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1" ||
  location.hostname.endsWith(".hf.space")
) ? "" : "https://MohamedGamal04-creditlens.hf.space";
