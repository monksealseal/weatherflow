import '@testing-library/jest-dom';
import { vi } from 'vitest';

global.URL.createObjectURL = vi.fn();

HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  getImageData: vi.fn(() => ({ data: [] })),
  putImageData: vi.fn(),
  createImageData: vi.fn(() => ({ data: [] })),
  setTransform: vi.fn(),
  resetTransform: vi.fn()
}));
