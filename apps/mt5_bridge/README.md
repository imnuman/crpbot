# MT5 Bridge Interface

Abstract interface for MT5/FTMO data access with fallback strategies.

## Current Implementation

- **Mock**: Default mock implementation for development/testing
- **Real**: Python MetaTrader5 module (to be implemented)

## Fallback Strategies

If the Python MetaTrader5 module has issues on Linux:

### Option 1: Windows VM + REST Shim

1. Install MT5 on a Windows VM
2. Create a simple REST API that exposes MT5 functions
3. Update `bridge.py` to use HTTP client instead of direct MT5

### Option 2: MetaAPI Provider

Use a third-party MetaAPI service that provides REST access to MT5 accounts.

## Implementation Notes

- The interface is abstract to allow easy provider swapping
- All implementations should handle connection errors gracefully
- Use read-only credentials for data collection (not trading account)

