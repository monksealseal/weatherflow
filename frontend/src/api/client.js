import axios from 'axios';
const baseURL = import.meta.env.VITE_API_URL || window.location.origin;
const client = axios.create({
    baseURL,
    headers: {
        'Content-Type': 'application/json'
    }
});
export async function fetchOptions() {
    const response = await client.get('/api/options');
    return response.data;
}
export async function runExperiment(config) {
    const response = await client.post('/api/experiments', config);
    return response.data;
}
