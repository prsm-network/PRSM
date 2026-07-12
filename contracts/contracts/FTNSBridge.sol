// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "./FTNSTokenSimple.sol";
import "./BridgeSecurity.sol";

/**
 * @title FTNSBridge
 * @dev Secure cross-chain bridge for FTNS tokens with multi-signature verification
 * 
 * Features:
 * - EIP-712 typed signature verification
 * - Configurable M-of-N validator threshold
 * - Replay attack protection
 * - Bridge in/out functionality
 * - Emergency controls
 * 
 * @author PRSM Protocol
 */
contract FTNSBridge is 
    Initializable,
    AccessControlUpgradeable,
    UUPSUpgradeable,
    ReentrancyGuardUpgradeable,
    PausableUpgradeable
{
    // ============ Role Definitions ============
    
    bytes32 public constant BRIDGE_ADMIN_ROLE = keccak256("BRIDGE_ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    
    // ============ State Variables ============
    
    /// @notice FTNS token contract
    FTNSTokenSimple public ftnsToken;
    
    /// @notice Bridge security contract for signature verification
    BridgeSecurity public bridgeSecurity;
    
    /// @notice Minimum bridge amount
    uint256 public minBridgeAmount;
    
    /// @notice Maximum bridge amount
    uint256 public maxBridgeAmount;
    
    /// @notice Bridge fee in basis points (100 = 1%)
    uint256 public bridgeFeeBps;
    
    /// @notice Fee recipient address
    address public feeRecipient;
    
    /// @notice Supported destination chains
    mapping(uint256 => bool) public supportedChains;
    
    /// @notice User nonces for bridge out operations
    mapping(address => uint256) public userBridgeOutNonce;
    
    /// @notice Processed source transactions (for bridge in)
    mapping(bytes32 => bool) public processedSourceTx;
    
    /// @notice Bridge statistics
    uint256 public totalBridgedOut;
    uint256 public totalBridgedIn;
    uint256 public totalFeesCollected;
    
    // ============ Events ============
    
    /**
     * @dev Emitted when tokens are bridged out
     */
    event BridgeOut(
        address indexed user,
        uint256 amount,
        uint256 fee,
        uint256 destinationChain,
        uint256 nonce,
        bytes32 indexed transactionId,
        bytes32 bridgeMessageHash
    );
    
    /**
     * @dev Emitted when tokens are bridged in
     */
    event BridgeIn(
        address indexed recipient,
        uint256 amount,
        uint256 sourceChain,
        bytes32 indexed sourceTxId,
        uint256 nonce,
        bytes32 indexed transactionId
    );
    
    /**
     * @dev Emitted when chain support is updated
     */
    event ChainSupportUpdated(uint256 indexed chainId, bool supported);
    
    /**
     * @dev Emitted when bridge limits are updated
     */
    event BridgeLimitsUpdated(uint256 minAmount, uint256 maxAmount);
    
    /**
     * @dev Emitted when bridge fee is updated
     */
    event BridgeFeeUpdated(uint256 feeBps, address feeRecipient);
    
    /**
     * @dev Emitted when bridge security contract is updated
     */
    event BridgeSecurityUpdated(address indexed oldSecurity, address indexed newSecurity);
    
    // ============ Errors ============
    
    error InvalidAmount();
    error InvalidChainId();
    error InvalidAddress();
    error ChainNotSupported();
    error InsufficientBalance();
    error InsufficientAllowance();
    error TransactionAlreadyProcessed();
    error SignatureVerificationFailed();
    error BridgeLimitExceeded();
    error BridgeLimitNotMet();
    error TransferFailed();
    
    // ============ Modifiers ============
    
    modifier onlyBridgeAdmin() {
        if (!hasRole(BRIDGE_ADMIN_ROLE, msg.sender)) {
            revert("FTNSBridge: caller is not bridge admin");
        }
        _;
    }
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    // ============ Initialization ============
    
    /**
     * @dev Initialize the bridge contract
     * @param admin Admin address
     * @param _ftnsToken FTNS token contract address
     * @param _bridgeSecurity Bridge security contract address
     * @param _feeRecipient Address to receive bridge fees
     * @param _minBridgeAmount Minimum bridge amount
     * @param _maxBridgeAmount Maximum bridge amount
     * @param _bridgeFeeBps Bridge fee in basis points
     */
    function initialize(
        address admin,
        address _ftnsToken,
        address _bridgeSecurity,
        address _feeRecipient,
        uint256 _minBridgeAmount,
        uint256 _maxBridgeAmount,
        uint256 _bridgeFeeBps
    ) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();
        __Pausable_init();
        
        if (admin == address(0) || _ftnsToken == address(0) || 
            _bridgeSecurity == address(0) || _feeRecipient == address(0)) {
            revert InvalidAddress();
        }
        
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(BRIDGE_ADMIN_ROLE, admin);
        _grantRole(OPERATOR_ROLE, admin);
        
        ftnsToken = FTNSTokenSimple(_ftnsToken);
        bridgeSecurity = BridgeSecurity(_bridgeSecurity);
        feeRecipient = _feeRecipient;
        minBridgeAmount = _minBridgeAmount;
        maxBridgeAmount = _maxBridgeAmount;
        bridgeFeeBps = _bridgeFeeBps;
        
        // Add common supported chains
        supportedChains[1] = true;     // Ethereum Mainnet
        supportedChains[137] = true;   // Polygon
        supportedChains[56] = true;    // BSC
        supportedChains[42161] = true; // Arbitrum
        supportedChains[10] = true;    // Optimism
    }
    
    // ============ Bridge Out Functions ============
    
    /**
     * @dev Bridge tokens out to another chain
     * @param amount Amount of tokens to bridge
     * @param destinationChain Target chain ID
     * @return transactionId Unique transaction identifier
     */
    function bridgeOut(
        uint256 amount,
        uint256 destinationChain
    ) external nonReentrant whenNotPaused returns (bytes32 transactionId) {
        // Validation
        if (amount < minBridgeAmount) revert BridgeLimitNotMet();
        if (amount > maxBridgeAmount) revert BridgeLimitExceeded();
        if (!supportedChains[destinationChain]) revert ChainNotSupported();
        
        // Check balance and allowance
        uint256 userBalance = ftnsToken.balanceOf(msg.sender);
        if (userBalance < amount) revert InsufficientBalance();
        
        uint256 allowance = ftnsToken.allowance(msg.sender, address(this));
        if (allowance < amount) revert InsufficientAllowance();
        
        // Calculate fee
        uint256 fee = (amount * bridgeFeeBps) / 10000;
        uint256 bridgeAmount = amount - fee;
        
        // Increment nonce
        uint256 nonce = ++userBridgeOutNonce[msg.sender];
        
        // Generate transaction ID
        transactionId = keccak256(abi.encodePacked(
            msg.sender,
            amount,
            destinationChain,
            nonce,
            block.timestamp,
            block.chainid
        ));
        
        // Generate bridge message hash for validators to sign on destination
        bytes32 bridgeMessageHash = keccak256(abi.encodePacked(
            "\x19\x01",
            bridgeSecurity.domainSeparator(),
            keccak256(abi.encode(
                bridgeSecurity.BRIDGE_MESSAGE_TYPEHASH_V2(),
                msg.sender,
                bridgeAmount,
                block.chainid,
                transactionId,
                nonce
            ))
        ));
        
        // Transfer tokens from user. sp1433 — CHECK the return value: a bridge must never proceed
        // to burn/mint as if it received tokens when it did not. FTNSTokenSimple's transferFrom
        // reverts on failure today, but a bridge is the canonical place a non-reverting-ERC20 swap
        // becomes a mint-from-nothing fund-loss, so the check is explicit (and it removes the real
        // Slither unchecked-transfer finding rather than baselining it).
        if (!ftnsToken.transferFrom(msg.sender, address(this), amount)) {
            revert TransferFailed();
        }
        
        // Burn the tokens (they will be minted on destination chain)
        ftnsToken.burnFrom(address(this), amount);
        
        // Mint fee to recipient if any
        if (fee > 0) {
            ftnsToken.mintReward(feeRecipient, fee);
            totalFeesCollected += fee;
        }
        
        // Update statistics
        totalBridgedOut += bridgeAmount;
        
        emit BridgeOut(
            msg.sender,
            bridgeAmount,
            fee,
            destinationChain,
            nonce,
            transactionId,
            bridgeMessageHash
        );
        
        return transactionId;
    }
    
    // ============ Bridge In Functions ============
    
    /**
     * @dev Bridge tokens in from another chain with multi-signature verification
     * @param message Bridge message containing transfer details
     * @param signatures Validator signatures
     * @return success Whether the bridge in was successful
     */
    function bridgeIn(
        BridgeSecurity.BridgeMessage calldata message,
        BridgeSecurity.ValidatorSignature[] calldata signatures
    ) external nonReentrant whenNotPaused returns (bool success) {
        // Basic validation
        if (message.recipient == address(0)) revert InvalidAddress();
        if (message.amount == 0) revert InvalidAmount();
        if (!supportedChains[message.sourceChainId]) revert ChainNotSupported();
        if (processedSourceTx[message.sourceTxId]) revert TransactionAlreadyProcessed();
        
        // Verify multi-signature through BridgeSecurity
        (bytes32 messageHash, bool isValid) = bridgeSecurity.verifyBridgeSignatures(
            message,
            signatures
        );
        
        if (!isValid) revert SignatureVerificationFailed();
        
        // Mark source transaction as processed
        processedSourceTx[message.sourceTxId] = true;
        
        // Mint tokens to recipient
        ftnsToken.mintReward(message.recipient, message.amount);
        
        // Update statistics
        totalBridgedIn += message.amount;
        
        // Generate local transaction ID
        bytes32 transactionId = keccak256(abi.encodePacked(
            message.recipient,
            message.amount,
            message.sourceChainId,
            message.sourceTxId,
            block.timestamp
        ));
        
        emit BridgeIn(
            message.recipient,
            message.amount,
            message.sourceChainId,
            message.sourceTxId,
            message.nonce,
            transactionId
        );
        
        return true;
    }
    
    /**
     * @dev Check if a bridge in would succeed without executing
     * @param message Bridge message to check
     * @param signatures Validator signatures
     * @return isValid Whether the signatures are valid
     * @return validCount Number of valid signatures
     */
    function checkBridgeIn(
        BridgeSecurity.BridgeMessage calldata message,
        BridgeSecurity.ValidatorSignature[] calldata signatures
    ) external view returns (bool isValid, uint256 validCount) {
        if (message.recipient == address(0) || message.amount == 0) {
            return (false, 0);
        }
        
        if (!supportedChains[message.sourceChainId]) {
            return (false, 0);
        }
        
        if (processedSourceTx[message.sourceTxId]) {
            return (false, 0);
        }
        
        return bridgeSecurity.checkBridgeSignatures(message, signatures);
    }
    
    // ============ Admin Functions ============
    
    /**
     * @dev Update supported chain
     * @param chainId Chain ID to update
     * @param supported Whether the chain is supported
     */
    function setChainSupport(uint256 chainId, bool supported) external onlyBridgeAdmin {
        supportedChains[chainId] = supported;
        emit ChainSupportUpdated(chainId, supported);
    }
    
    /**
     * @dev Update bridge limits
     * @param _minBridgeAmount New minimum bridge amount
     * @param _maxBridgeAmount New maximum bridge amount
     */
    function setBridgeLimits(
        uint256 _minBridgeAmount,
        uint256 _maxBridgeAmount
    ) external onlyBridgeAdmin {
        if (_minBridgeAmount == 0 || _maxBridgeAmount <= _minBridgeAmount) {
            revert InvalidAmount();
        }
        
        minBridgeAmount = _minBridgeAmount;
        maxBridgeAmount = _maxBridgeAmount;
        
        emit BridgeLimitsUpdated(_minBridgeAmount, _maxBridgeAmount);
    }
    
    /**
     * @dev Update bridge fee
     * @param _bridgeFeeBps New fee in basis points
     * @param _feeRecipient New fee recipient
     */
    function setBridgeFee(
        uint256 _bridgeFeeBps,
        address _feeRecipient
    ) external onlyBridgeAdmin {
        if (_bridgeFeeBps > 1000) revert("Fee cannot exceed 10%");
        if (_feeRecipient == address(0)) revert InvalidAddress();
        
        bridgeFeeBps = _bridgeFeeBps;
        feeRecipient = _feeRecipient;
        
        emit BridgeFeeUpdated(_bridgeFeeBps, _feeRecipient);
    }
    
    /**
     * @dev Update bridge security contract
     * @param _bridgeSecurity New bridge security contract address
     */
    function setBridgeSecurity(address _bridgeSecurity) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_bridgeSecurity == address(0)) revert InvalidAddress();
        
        address oldSecurity = address(bridgeSecurity);
        bridgeSecurity = BridgeSecurity(_bridgeSecurity);
        
        emit BridgeSecurityUpdated(oldSecurity, _bridgeSecurity);
    }
    
    /**
     * @dev Pause bridge operations
     */
    function pause() external onlyBridgeAdmin {
        _pause();
    }
    
    /**
     * @dev Unpause bridge operations
     */
    function unpause() external onlyBridgeAdmin {
        _unpause();
    }
    
    // ============ View Functions ============
    
    /**
     * @dev Get bridge statistics
     * @return bridgedOut Total tokens bridged out
     * @return bridgedIn Total tokens bridged in
     * @return fees Total fees collected
     * @return netFlow Net flow (positive = more in, negative = more out)
     */
    function getBridgeStats() external view returns (
        uint256 bridgedOut,
        uint256 bridgedIn,
        uint256 fees,
        int256 netFlow
    ) {
        bridgedOut = totalBridgedOut;
        bridgedIn = totalBridgedIn;
        fees = totalFeesCollected;
        
        if (totalBridgedIn >= totalBridgedOut) {
            netFlow = int256(totalBridgedIn - totalBridgedOut);
        } else {
            netFlow = -int256(totalBridgedOut - totalBridgedIn);
        }
    }
    
    /**
     * @dev Check if a source transaction has been processed
     * @param sourceTxId Source transaction ID
     * @return Whether the transaction has been processed
     */
    function isSourceTxProcessed(bytes32 sourceTxId) external view returns (bool) {
        return processedSourceTx[sourceTxId];
    }
    
    /**
     * @dev Get user's next bridge out nonce
     * @param user User address
     * @return Next nonce for bridge out
     */
    function getUserNextNonce(address user) external view returns (uint256) {
        return userBridgeOutNonce[user] + 1;
    }
    
    /**
     * @dev Compute the bridge message hash for off-chain signing
     * @param message Bridge message
     * @return The EIP-712 typed hash
     */
    function computeBridgeMessageHash(
        BridgeSecurity.BridgeMessage calldata message
    ) external view returns (bytes32) {
        return bridgeSecurity.hashBridgeMessage(message);
    }
    
    // ============ Internal Functions ============
    
    /**
     * @dev Required override for UUPS upgradeability
     */
    function _authorizeUpgrade(address newImplementation)
        internal
        override
        onlyRole(DEFAULT_ADMIN_ROLE)
    {}
    
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(AccessControlUpgradeable)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
