import { LEGAL_API_BASE_URL } from '$lib/constants';

export const legalSearch = async (token: string, query: string, resultsLimit: string) => {
    try {
        const response = await fetch(`${LEGAL_API_BASE_URL}/legalsearch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${token}`
            },
            body: JSON.stringify({ query, results_limit: resultsLimit })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (err) {
        console.error("Fetch error:", err);
        throw new Error(`Error performing legal search: ${err.message}`);
    }
};
