// src/lib/apis/docgen/index.ts
import { DOCGEN_API_BASE_URL } from '$lib/constants';

export const docgen = async (base_prompt:string, step_prompt: string) => {
    let error = null;
    console.log("test1");
    const response = await fetch(`${DOCGEN_API_BASE_URL}/docgen`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
	    
        },
        body: JSON.stringify({ "base_prompt": base_prompt, "step_prompt": step_prompt })
    }).then(res => {
	console.log("sent");
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    }).catch(err => {
        console.error("Fetch error:", err);
        error = err;
    });

    if (error) {
        throw new Error("Error performing document generation");
    }

    return response;
};
