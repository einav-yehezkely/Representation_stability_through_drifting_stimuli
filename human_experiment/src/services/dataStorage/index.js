// Storage composition: the only place that knows concrete adapters. A future server/Supabase
// adapter is added here and selected through config.storage.adapter; phase logic, stimulus
// selection, timing and response coding do not change.

import { browserDownload } from "./browserDownload.js";
import { LocalCsvStorage } from "./localCsvStorage.js";
import { MemoryStorage } from "./memoryStorage.js";
import { assertImplementsContract } from "./storageContract.js";

export function createStorage(config, overrides = {}) {
  let adapter;
  switch (config.storage.adapter) {
    case "localCsv":
      adapter = new LocalCsvStorage({
        prefix: config.storage.localStoragePrefix,
        download: browserDownload,
        ...overrides,
      });
      break;
    case "memory":
      adapter = new MemoryStorage(overrides);
      break;
    default:
      throw new Error(`Unsupported storage adapter: ${config.storage.adapter}`);
  }
  return assertImplementsContract(adapter);
}

export { StorageError, STORAGE_ERROR_CODES } from "./storageContract.js";
